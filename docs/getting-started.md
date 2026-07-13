# Getting started

← [back to the README](../README.md)

Three things have to be in place before your first run: one system tool, a Python environment, and the model weights. The weights are the annoying one, so let's be honest about it up front.

---

## 1. Install `cmake`

`dlib` — the library that finds and straightens faces — builds from source, and it needs `cmake`.

```bash
brew install cmake        # macOS
apt install cmake         # Debian/Ubuntu
```

## 2. Create the environment

Either path works. The venv one is what actually gets used day to day.

```bash
# plain venv (uv or pip)
uv venv --python 3.13 && uv pip install -r requirements.txt
source .venv/bin/activate

# ...or conda, if you prefer
conda env create -n pulse -f pulse.yml
conda activate pulse
```

The device — CUDA, Apple Silicon (MPS), or CPU — is auto-detected. You don't configure anything. Force one with `PULSE_DEVICE=cpu` (or `mps`/`cuda`) if you need to.

Both files pin **`torch>=2.10,<2.13`**, and that upper bound is not laziness. torch 2.13.0 quietly regressed the MPS backward pass by ~25% on this model. The forward pass is untouched, so nothing looks broken — it's just slower, invisibly, unless you profile. [The full story](apple-silicon.md#the-torch-213-trap).

## 3. Get the model weights

Here's the part nobody enjoys. **The original download links are dead.** Google Drive, and a later mirror — both gone. The code will look for these files in the repo root and use them if they're there:

| file                                    | needed?                  | what it is                                                                                                                                  |
| --------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthesis.pt`                          | **yes**                  | The StyleGAN generator. This is the actual "brain" — without it nothing runs.                                                               |
| `gaussian_fit.pt`                       | already here             | Committed to the repo. Statistics of the mapping network's output, precomputed.                                                             |
| `mapping.pt`                            | not really               | Only used to regenerate `gaussian_fit.pt`, which already exists. You can skip it.                                                           |
| `shape_predictor_68_face_landmarks.dat` | only for `align_face.py` | The stock dlib landmark model — this one is still downloadable: [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |

So in practice you need **`synthesis.pt`** to super-resolve anything, plus the dlib file if you want to preprocess your own photos. `synthesis.pt` and `mapping.pt` are PULSE's repackaged StyleGAN CelebA-HQ weights; you'll have to source them yourself. (If you host them somewhere, point the URLs in `PULSE.py` and `align_face.py` at your host and auto-download works again.)

---

## Preparing your photos

`run.py` reads `./input/` and expects each image to already be a **square, face-aligned, power-of-two PNG**. If your photos aren't in that shape yet, `align_face.py` does it for you: it finds the face, straightens it, crops it, and shrinks it.

```bash
python align_face.py -input_dir realpics -output_dir input -output_size 32
```

**One gotcha worth internalizing:** if a photo is _already_ low-resolution, shrinking it further throws away most of what little is left. Bicubically upscale it to 1024×1024 first and let `align_face.py` do the downscaling. You want the _alignment_ from a good source image, not a smaller version of an already-bad one.

The original authors tested on [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), but in their experience it works on more or less any photo of a realistic face.

## Running it

```bash
python run.py
```

That's it. Every `*.png` in `input/` becomes a 1024×1024 face in `runs/`.

Watch the output for this:

```
NOT CONVERGED: best L2 0.00670 > eps 0.002 after 100 steps (still improving). Raise -steps (try 200).
```

It means the search ran out of steps while it was still making progress — the face you got back does **not** downscale onto your input as closely as you asked. It isn't a crash and it isn't a warning you can ignore; it's the tool telling you the answer is unfinished. Raise `-steps` until it stops saying it. See [Tuning](tuning.md) for what that's worth (a lot).

---

## Command-line reference

These tables are generated from the `argparse` definitions; `python run.py -h` prints the same thing.

### `run.py`

| Argument                      | Default                | Description                                                                                                                                            |
| ----------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `-input_dir`                  | `input`                | input data directory                                                                                                                                   |
| `-output_dir`                 | `runs`                 | output data directory                                                                                                                                  |
| `-cache_dir`                  | `cache`                | cache directory for model weights                                                                                                                      |
| `-duplicates`                 | `1`                    | how many HR images to produce for every input image                                                                                                    |
| `-batch_size`                 | `1`                    | batch size during optimization (leave it at 1 — see [Apple Silicon notes](apple-silicon.md#two-levers-that-do-not-work))                               |
| `-seed`                       | _(none)_               | manual seed, for reproducibility                                                                                                                       |
| `-loss_str`                   | `100*L2+0.05*GEOCROSS` | the objective, as weighted terms (`L2`, `L1`, `GEOCROSS`)                                                                                              |
| `-eps`                        | `2e-3`                 | "good enough" floor on the downscaling loss                                                                                                            |
| `-noise_type`                 | `trainable`            | `zero`, `fixed`, or `trainable`                                                                                                                        |
| `-num_trainable_noise_layers` | `5`                    | how many noise layers to optimize                                                                                                                      |
| `-tile_latent`                | `False` (flag)         | tile one latent across all 18 layers (W instead of W+)                                                                                                 |
| `-bad_noise_layers`           | `17`                   | noise layers forced to zero. **Split on `.`** (e.g. `3.5`), not commas                                                                                 |
| `-opt_name`                   | `adam`                 | `sgd`, `adam`, `sgdm`, or `adamax`                                                                                                                     |
| `-learning_rate`              | `0.4`                  | step size                                                                                                                                              |
| `-steps`                      | `100`                  | **the one that matters most — the default is too low for hard inputs. [See Tuning](tuning.md)**                                                        |
| `-lr_schedule`                | `linear1cycledrop`     | `fixed`, `linear1cycledrop`, or `linear1cycle`                                                                                                         |
| `-save_intermediate`          | `False` (flag)         | write per-step HR/LR frames into `runs/<name>/{HR,LR}/`                                                                                                |
| `-compile`                    | `False` (flag)         | `torch.compile` the generator. ~11% faster per step, but the warmup makes it a net **loss** under ~200 steps. [Details](apple-silicon.md#torchcompile) |
| `-precision`                  | `fp32`                 | `fp32`, `mixed`, or `fp16`. [Details](apple-silicon.md#half-precision)                                                                                 |

### `align_face.py`

| Argument       | Default    | Description                            |
| -------------- | ---------- | -------------------------------------- |
| `-input_dir`   | `realpics` | directory of unprocessed photos        |
| `-output_dir`  | `input`    | where the aligned crops go             |
| `-output_size` | `32`       | downscale target; must be a power of 2 |
| `-seed`        | _(none)_   | manual seed                            |
| `-cache_dir`   | `cache`    | cache directory for model weights      |

---

**Next:** [How it works](how-it-works.md) · [Tuning](tuning.md) · [Apple Silicon notes](apple-silicon.md)
