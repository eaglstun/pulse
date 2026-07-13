# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PULSE (CVPR'20) does self-supervised photo super-resolution. Instead of training a network to map LR→HR, it freezes a pretrained StyleGAN generator and runs **projected gradient descent over the generator's latent space** to find a high-res face that, when bicubically downscaled, matches the low-res input. The output is an imagined realistic face that downscales correctly — not a reconstruction of any real person.

## Environment & running

Runs on **CUDA, Apple MPS (Metal), or CPU** — `device.py::get_device()` auto-selects in that order and every module imports the shared `device` from it. Force a specific backend with the `PULSE_DEVICE` env var (e.g. `PULSE_DEVICE=cpu python run.py`), which is the escape hatch if an MPS op is unsupported on your torch build. `cmake` must be installed before creating the env (needed to build `dlib`).

Two equivalent env paths: **`requirements.txt`** (plain venv — this is what's actually used here; `.venv/` exists in the repo root) and **`pulse.yml`** (conda). Both pin `torch>=2.10,<2.13`, `torchvision>=0.25,<0.28`, plus `numpy scipy pillow requests dlib`.

```bash
uv venv --python 3.13 && uv pip install -r requirements.txt   # cmake must be on PATH first (dlib)
.venv/bin/python run.py
```

**The `torch<2.13` cap is deliberate — do not lift it.** torch 2.13.0 regressed the MPS _backward_ pass ~25% on this model (the forward is untouched, so it's invisible without profiling): 2.13 moved `sum` off MPSGraph onto a Metal kernel that gives one 32-lane SIMD group per _output_ element, which collapses when the reduced dim is small — precisely the shape an `expand`/broadcast backward produces. Measured: 22.4 it/s on 2.10–2.12 vs 16.7 on 2.13. A fix exists on Eric's pytorch fork (`mps-sum-small-reduced-extent`, `3102913a757`) and restores full parity, but it is **not in any released wheel yet**. Lift the cap only when a PyTorch _release_ contains it, and re-benchmark first.

Two-step pipeline:

```bash
# 1. (optional) align + downscale raw photos: realpics/ -> input/
python align_face.py -input_dir realpics -output_dir input -output_size 32

# 2. super-resolve every *.png in input/ -> runs/
python run.py
```

Skip step 1 only if `input/` already holds square, aligned, power-of-2-sized face PNGs. All directories and hyperparameters are CLI args; the full annotated list lives at the top of `run.py` and `align_face.py`. There is **no test suite, linter, or build step** — this is a research script.

Common `run.py` knobs: `-duplicates N` (produce N variant HR images per input), `-steps`, `-learning_rate`, `-eps` (target downscaling-loss threshold), `-loss_str`, `-save_intermediate` (write per-step HR/LR frames into `runs/<name>/{HR,LR}/`), `-compile` (torch.compile the generator), `-precision {fp32,mixed,fp16}`.

## ⚠️ `-steps` is the highest-leverage parameter, and the default (100) is too low

The single most important thing to know before optimizing anything here. Measured on an M4 Max, scored on the downscaling L2 of the saved image (PULSE's own objective):

- `demo3` (hard input): **halves its error** at 200 steps vs the default 100.
- `demo5` (hard input): needs **~800** (−70%).
- `demo` (easy input): converged by ~step 30; more steps gain nothing.

**Every kernel-level optimization on this repo, combined, is worth ~8%. Changing that one number is worth 50%.** Perf work here matters because it makes steps _affordable_ — it was never the prize. GEOCROSS improves alongside L2, so the extra steps are not trading realism for pixel accuracy.

`PULSE.forward()` prints `NOT CONVERGED: ... Raise -steps` when a run never reaches `-eps`. `loss.py::_loss_l2` **clamps at eps**, so `min_l2` bottoms out _at_ eps and can never go below — sitting on that floor _is_ convergence, and any convergence check needs a tolerance (`> eps*1.01`) or float noise at exactly eps reads as failure.

## Perf: two levers that do NOT work on MPS

Don't re-try these; both were measured.

- **`-batch_size > 1` buys nothing.** Throughput is flat from batch 1 to 8 (20.9 → 21.7 img/s) — one 1024×1024 pass already saturates the GPU.
- **`channels_last` is a regression** (22.2 → 15.1 it/s, 0.68×), despite being the CUDA reflex.

Also: **MPS dispatches asynchronously.** A wall-clock timer that doesn't sync first measures the _enqueue_ rate, not compute, and will report impossible throughput. Use `device.py::sync_device()` before stopping any clock. And when benchmarking on Eric's machine, check `pgrep -f "fa[-]mps-venv"` first — his flash-attention MPS suite contends for the GPU and silently wrecks timings.

## Model weights & caching

Three weight files live in the repo root (gitignored, not tracked): `synthesis.pt`, `mapping.pt`, `shape_predictor_68_face_landmarks.dat`. `PULSE.py` and `align_face.py` **load these local files directly if present**, and only fall back to downloading when they're missing. The fallback download URLs (`https://ericeaglstun.com/misc/...`, and the original Google Drive links above them, both kept as commented context) are **no longer live** — so a fresh clone without the weights won't run until they're supplied. `shape_predictor_68_face_landmarks.dat` is the stock dlib model (`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`); `synthesis.pt`/`mapping.pt` are PULSE's repackaged StyleGAN CelebA-HQ weights. `drive.py::open_url` handles the download-path caching (MD5-keyed into `cache/`).

`gaussian_fit.pt` (committed to the repo) stores the mean/std of the mapping network's output over 1M samples. If absent, `PULSE.__init__` loads the mapping network (`mapping.pt`) and regenerates it — an expensive one-time step. Because this is committed, `mapping.pt` is **not needed at runtime** in practice; only `synthesis.pt` is. Don't delete `gaussian_fit.pt` unless you intend to recompute.

## Architecture / control flow

The optimization loop is the heart of the system — `PULSE.forward()` in `PULSE.py`:

1. **Latent + noise are the optimized variables**, not network weights. The generator is frozen (`requires_grad = False`). A `(batch, 18, 512)` latent plus up to `num_trainable_noise_layers` StyleGAN noise tensors form the parameter set.
2. **Latent re-parameterization**: the raw latent is affine-mapped by the cached `gaussian_fit` mean/std then passed through LeakyReLU, so optimizing a standard-normal variable lands in the mapping network's actual output distribution.
3. **`SphericalOptimizer`** (wraps Adam/SGD/etc.) projects each parameter back onto the hypersphere of its initial radius after every step — this is the geometric prior central to the paper.
4. **`LossBuilder`** parses `loss_str` like `"100*L2+0.05*GEOCROSS"` into weighted terms. `L2`/`L1` measure downscaled-output vs reference (clamped at `eps` so it stops once "close enough"); `GEOCROSS` is a geodesic spread penalty across the 18 latent vectors. The downscaler `D` is `BicubicDownSample` — a fixed, **differentiable** bicubic conv (gradients flow through it, which is what makes the whole scheme work).
5. `forward()` is a **generator that `yield`s** `(HR, LR)` tensors — the **best** iterate always (by TOTAL loss, tracked on-device with `torch.where`), plus every intermediate step when `-save_intermediate`. `run.py` consumes the generator and writes PNGs. The loss is **not monotone** (the 1cycle schedule re-raises the LR mid-run), so the best step is often not the last — don't "simplify" this back to yielding the final `gen_im`; that was a six-year-old bug, fixed in `7618167`.
6. **The loop never syncs to the host.** Tracking the best iterate with a Python `if loss < min_loss` forces a GPU→CPU sync every step, which drains the Metal command queue. Everything stays on-device and is read back once, after the loop (and `sync_device()` is called before the timer stops — see above).
7. **Optional half precision** (`-precision mixed|fp16`): the frozen generator runs in fp16 while the optimized latent/noise and the loss stay fp32. `mixed` casts only the 256/512/1024 blocks + toRGB, once on entry to that contiguous region — _not_ per block, which would round-trip a 1024² activation four times and give back half the speedup. Counter-intuitively the big late blocks are the _most_ fp16-tolerant and the tiny early ones the least; see the README.

`run.py` wraps the model in `torch.nn.DataParallel` **only for multi-GPU CUDA** (single-GPU/MPS/CPU skip it). Either way `forward()` receives a `**kwargs` dict and pulls named args out of it, and moves `ref_im` onto the target device itself (under DataParallel the input is already scattered onto the GPU; otherwise `forward` does the `.to(device)`).

### File map

- `PULSE.py` — the optimization model (the core).
- `run.py` — entry point: dataset, dataloader, CLI, output writing.
- `stylegan.py` — `G_mapping` + `G_synthesis` StyleGAN modules (adapted from lernapparat/NVlabs); the frozen generator.
- `loss.py` — `LossBuilder`, the weighted-loss parser and loss terms.
- `bicubic.py` — `BicubicDownSample`, the differentiable downscaler used by both the loss and `align_face.py`.
- `SphericalOptimizer.py` — radius-preserving optimizer wrapper.
- `align_face.py` + `shape_predictor.py` — dlib-based face landmark detection, alignment, and crop (preprocessing).
- `drive.py` — URL download + caching helper.
- `device.py` — `get_device()` (CUDA → MPS → CPU, `PULSE_DEVICE` override) and `sync_device()` (blocks until the queued GPU work actually finishes; call before stopping any timer).

## Conventions

- Images use NCHW float tensors in `[0, 1]`. StyleGAN emits `[-1, 1]`, normalized to `[0, 1]` via `(x+1)/2` right after `self.synthesis(...)`.
- All spatial dims are square and powers of 2; HR is fixed at 1024×1024, so `factor = 1024 // input_size` must divide evenly (asserted in `loss.py` and `align_face.py`).
- `bad_noise_layers` is parsed splitting on `.` (e.g. `"17"`, or `"3.5"`), not commas — note this if changing the default.
