# PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models

Code accompanying CVPR'20 paper of the same title. Paper: [arXiv:2003.03808](https://arxiv.org/abs/2003.03808) ([CVPR open-access PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.pdf)). A Markdown transcription of the paper is in [`paper.md`](paper.md).

## NOTE

The original authors noted concern that PULSE would be used to identify individuals whose faces have been blurred out, and emphasized that this is impossible - **PULSE makes imaginary faces of people who do not exist, which should not be confused for real people.** It will **not** help identify or reconstruct the original image.

They also addressed concerns of bias in PULSE: **the [paper](https://arxiv.org/abs/2003.03808) includes a section, along with an accompanying model card, directly addressing this bias.**

---

## Fork notes

This fork updates the original CVPR'20 code so it runs on modern hardware and without the (now-defunct) hosted model weights:

- **Runs on CUDA, Apple Silicon (MPS/Metal), or CPU** — the device is auto-selected by `device.py`; no code changes needed. Force one with `PULSE_DEVICE=cpu` (or `mps`/`cuda`).
- **Loads model weights from local files** (`synthesis.pt`, `mapping.pt`, `shape_predictor_68_face_landmarks.dat`) when present in the repo root, falling back to download only if they're missing. The original Google Drive and self-hosted download links are no longer live, so place these files in the repo root yourself. `shape_predictor_68_face_landmarks.dat` is the stock dlib model (`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`).
- `pulse.yml` (conda) and `requirements.txt` (plain venv) are modernized to PyTorch 2.x, Python 3.13, and only the packages the code actually uses.
- **`torch` is capped below 2.13 on purpose.** torch 2.13.0 regressed the MPS _backward_ pass by ~25% on this model — the forward pass is completely unaffected, so it's invisible unless you profile. Root cause: 2.13 moved `sum` off MPSGraph onto a Metal kernel that assigns one 32-lane SIMD group per _output_ element, which collapses when the reduced dimension is small — exactly the shape produced by the backward of an `expand`/broadcast. Measured on an M4 Max: 22.4 it/s on torch 2.10–2.12 vs 16.7 on 2.13. `torch.compile` sidesteps it.
- **Saves the _best_ iterate, not the last one.** The final yield used to return the last step's image as HR while taking LR from the best step — two different images, and neither matched the `BEST (n)` printed above it. That was an accident introduced in 2020 when `forward()` became a generator; the pre-2020 code returned `best_im`.
- **Faster on Apple Silicon** — a handful of MPS-specific fixes worth ~8% in total (see [Performance notes](#performance-notes-apple-silicon)). Worth far less than raising `-steps`, which is the point of that section.

![Transformation Preview](./readme_resources/014.jpeg)
![Transformation Preview](./readme_resources/034.jpeg)
![Transformation Preview](./readme_resources/094.jpeg)

# Table of Contents

- [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models](#pulse-self-supervised-photo-upsampling-via-latent-space-exploration-of-generative-models)
- [Table of Contents](#table-of-contents)
  - [Fork notes](#fork-notes)
  - [What does it do?](#what-does-it-do)
  - [Usage](#usage)
    - [Prereqs](#prereqs)
    - [Data](#data)
    - [Applying PULSE](#applying-pulse)
    - [Command-line reference](#command-line-reference)
    - [What is latent space?](#what-is-latent-space)
    - [Understanding the key parameters](#understanding-the-key-parameters)
    - [`-steps` is the parameter that matters most (and the default is too low)](#-steps-is-the-parameter-that-matters-most-and-the-default-is-too-low)
    - [Performance notes (Apple Silicon)](#performance-notes-apple-silicon)
    - [Half precision](#half-precision)
    - [Parameter sweep (worked example)](#parameter-sweep-worked-example)

## What does it do?

In plain terms: you give PULSE a small, blurry photo of a face, and it invents a sharp, realistic face that — when shrunk back down to the size of your blurry one — looks just like your input. It does not _recover_ detail that was lost in the blur (that information is gone for good); it _imagines_ a believable high-res face that is consistent with the blur. That is also why two runs can hand you two different people who both shrink down to the same photo.

More precisely: given a low-resolution input image, PULSE searches the outputs of a generative model (here, [StyleGAN](https://github.com/NVlabs/stylegan)) for high-resolution images that are perceptually realistic and downscale correctly.

![Transformation Preview](./readme_resources/transformation.gif)

## Usage

The main file of interest for applying PULSE is `run.py`. A full list of arguments with descriptions can be found in that file; the ones relevant to getting started are described below.

### Prereqs

Before your first run you need three things in place: one system tool, the Python environment, and the pretrained model files.

1. **Install `cmake`.** It's needed to build `dlib`, the library that finds and straightens faces (on a Mac: `brew install cmake`; on Debian/Ubuntu: `apt install cmake`).
2. **Create the Python environment.** This fork runs on an NVIDIA GPU (CUDA), Apple Silicon (MPS), or a plain CPU — the right one is picked automatically (see [Fork notes](#fork-notes)); the original only supported CUDA and was tested on Linux and Windows. Either path works:

   ```bash
   # plain venv (uv or pip)
   uv venv --python 3.13 && uv pip install -r requirements.txt
   source .venv/bin/activate

   # ...or conda, if you prefer
   conda env create -n pulse -f pulse.yml
   conda activate pulse
   ```

   Both pin **`torch>=2.10,<2.13`**, and the upper bound is deliberate: torch 2.13.0 regressed the MPS _backward_ pass by ~25% on this model. The forward pass is untouched, so the damage is invisible until you profile. See [Fork notes](#fork-notes).

3. **Put the pretrained model files in the repo root:** `synthesis.pt`, `mapping.pt`, and `shape_predictor_68_face_landmarks.dat`. These are the trained "brain" of the model — without them nothing can run — and the code loads them directly when they're present. The original auto-download links (Google Drive, and a later mirror) are no longer live, so you'll need to obtain the files yourself. `shape_predictor_68_face_landmarks.dat` is the stock dlib model from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`. (To restore auto-download, edit the URLs in `PULSE.py` and `align_face.py` to point at a host you control.)

### Data

`run.py` reads its inputs from `./input/` by default, and it expects each image to already be a square, face-aligned, low-resolution PNG. If your photos aren't in that form yet, don't worry — put the originals in `realpics/` and run `align_face.py` first. It locates the face, crops and straightens it, and shrinks it down for you (you pick the downscaling factor at this step). As before, any of these directories can be changed with a command-line argument.

One gotcha: if a photo is _already_ low-resolution, shrinking it further throws away almost all the detail that's left. In that case, bicubically upscale it first (usually to 1024×1024) and let `align_face.py` handle the downscaling.

The original authors tested on the [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) face dataset, but in their experience PULSE works on just about any photo of a realistic face.

### Applying PULSE

Once your data is appropriately formatted, all you need to do is

```bash
python run.py
```

Enjoy!

> _Any resemblance to actual persons, living or dead, is purely coincidental — and also mathematically the whole point._

### Command-line reference

All directories and hyperparameters are command-line arguments. The tables below are generated from the `argparse` definitions at the top of each script (`python run.py -h` / `python align_face.py -h` print the same descriptions).

**`run.py`** — super-resolve every `*.png` in the input dir:

| Argument                      | Default                | Description                                                                                                                                                                                               |
| ----------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-input_dir`                  | `input`                | input data directory                                                                                                                                                                                      |
| `-output_dir`                 | `runs`                 | output data directory                                                                                                                                                                                     |
| `-cache_dir`                  | `cache`                | cache directory for model weights                                                                                                                                                                         |
| `-duplicates`                 | `1`                    | how many HR images to produce for every image in the input directory                                                                                                                                      |
| `-batch_size`                 | `1`                    | batch size to use during optimization                                                                                                                                                                     |
| `-seed`                       | _(none)_               | manual seed to use                                                                                                                                                                                        |
| `-loss_str`                   | `100*L2+0.05*GEOCROSS` | loss function to use (weighted terms: `L2`, `L1`, `GEOCROSS`)                                                                                                                                             |
| `-eps`                        | `2e-3`                 | target for the downscaling loss (L2); optimization stops contributing once within this                                                                                                                    |
| `-noise_type`                 | `trainable`            | `zero`, `fixed`, or `trainable`                                                                                                                                                                           |
| `-num_trainable_noise_layers` | `5`                    | number of noise layers to optimize                                                                                                                                                                        |
| `-tile_latent`                | `False` (flag)         | forcibly tile the same latent 18 times                                                                                                                                                                    |
| `-bad_noise_layers`           | `17`                   | noise layers to zero out to improve image quality (split on `.`, e.g. `3.5`)                                                                                                                              |
| `-opt_name`                   | `adam`                 | optimizer for projected gradient descent (`sgd`, `adam`, `sgdm`, `adamax`)                                                                                                                                |
| `-learning_rate`              | `0.4`                  | learning rate to use during optimization                                                                                                                                                                  |
| `-steps`                      | `100`                  | number of optimization steps                                                                                                                                                                              |
| `-lr_schedule`                | `linear1cycledrop`     | `fixed`, `linear1cycledrop`, or `linear1cycle`                                                                                                                                                            |
| `-save_intermediate`          | `False` (flag)         | store and save intermediate HR and LR images during optimization                                                                                                                                          |
| `-compile`                    | `False` (flag)         | `torch.compile` the synthesis network: ~11% faster per step, but a one-off warmup (~1s warm cache, ~7s cold) makes it a net **loss** at the default 100 steps. Wins past ~200 steps or across many images |
| `-precision`                  | `fp32`                 | generator precision: `fp32`, `mixed` (fp16 for the 256/512/1024 blocks only), or `fp16` (whole network). Latent, noise and loss stay fp32 regardless. See [Half precision](#half-precision)   |

**`align_face.py`** — (optional preprocessing) align + downscale raw photos:

| Argument       | Default    | Description                                                 |
| -------------- | ---------- | ----------------------------------------------------------- |
| `-input_dir`   | `realpics` | directory with unprocessed images                           |
| `-output_dir`  | `input`    | output directory                                            |
| `-output_size` | `32`       | size to downscale the input images to, must be a power of 2 |
| `-seed`        | _(none)_   | manual seed to use                                          |
| `-cache_dir`   | `cache`    | cache directory for model weights                           |

### What is latent space?

"Latent space" comes up constantly below, so here's the intuition. A generator like [StyleGAN](https://github.com/NVlabs/stylegan) is just a function that turns a short list of numbers into an image:

```text
latent vector  →  StyleGAN  →  image
 (512 numbers)                 (1024×1024 face)
```

The **latent space** is the space of all possible input vectors. Every point in it maps to some output face, and moving around in it moves you smoothly between faces. It's "latent" (hidden) because those numbers aren't pixels — they're an abstract code the network learned to expand into a full picture.

A useful analogy is a music synthesizer: you don't draw the sound wave directly, you turn a few dozen knobs and the synth produces audio. The space of all knob settings is its latent space. StyleGAN is the same idea for faces — hand it ~512 "knob" numbers and it renders a photorealistic face; nudge them and the face changes smoothly (older, smiling, different hair).

The reason this is useful is that a trained GAN's latent space is **organized**, not random:

- **Nearby points → similar images**, so you can interpolate and use gradient descent to slide toward a target.
- **Directions carry meaning** — one direction adds a smile, another rotates the head, another ages the face.
- **Realistic faces occupy a thin region** (a curved surface, or _manifold_) inside the space; points on it look like faces, points far off it look like noise.

This is exactly what PULSE exploits: it freezes StyleGAN and **searches the latent space** for an input whose output downscales onto your photo (relying on smoothness for the search), while constraining that search to stay on the realistic-face manifold (so the result is a believable face, not noise that happens to downscale right).

StyleGAN actually exposes a few latent spaces, which is what [`-tile_latent`](#understanding-the-key-parameters) toggles between: a raw **Z** vector, a more disentangled **W** vector (Z run through a mapping network), and **W+** — 18 separate W vectors, one per layer, where coarse layers control pose/shape and fine layers control texture/color. PULSE optimizes all 18 (W+) by default for maximum expressiveness; `-tile_latent` collapses them back to a single shared W.

### Understanding the key parameters

The table above says _what_ each flag is; this section explains _why_ the interesting ones exist. First, a mental model.

PULSE never "enhances" your photo directly. It holds a pretrained [StyleGAN](https://github.com/NVlabs/stylegan) generator **fixed** and **searches its latent space** — the space of all faces the GAN can draw — for a high-res face that, when bicubically shrunk back down, lands on your low-res input. The search is ordinary gradient descent, but over the generator's _input_ (a latent vector plus some noise) rather than over any network weights. Two competing forces pull on that search:

1. **"Downscale correctly"** — the generated face, shrunk, must match your LR image. (the `L2`/`L1` loss terms)
2. **"Stay a realistic face"** — the latent must stay in the region StyleGAN considers natural, so you get a believable face and not random noise that happens to downscale right. (the spherical constraint + the `GEOCROSS` term)

Most of the non-obvious flags are knobs on that balance.

- **`-loss_str`** — the objective itself, written as `weight*TERM+weight*TERM`. The default `100*L2+0.05*GEOCROSS` means "mostly match the downscaled pixels, with a light realism nudge." Terms:
  - **`L2` / `L1`** — distance between the **downscaled** generated image and your input — the "downscales correctly" force. `L1` tolerates outlier pixels more than `L2`.
  - **`GEOCROSS`** — a _geodesic cross_ penalty. The latent is really 18 separate 512-d style vectors (one per StyleGAN layer); `GEOCROSS` measures how far apart they sit on the sphere and penalizes spreading them out. Keeping them together ties the result to StyleGAN's natural faces (more realistic); a smaller weight allows more per-layer detail but risks artifacts.
- **`-eps`** (default `2e-3`) — a "good enough" floor on the downscaling loss. Once the per-image match is within `eps`, that term stops pushing, so the optimizer doesn't keep chasing exact LR pixels at the expense of realism. Lower = stricter match, higher = more creative license.
- **`-noise_type`** + **`-num_trainable_noise_layers`** — StyleGAN injects tiny per-pixel "noise" at each layer to render fine texture (pores, stray hairs). `zero` ignores it, `fixed` picks a random texture and freezes it, `trainable` optimizes it too. With `trainable`, `-num_trainable_noise_layers` (default 5) sets how many layers — coarsest first — get optimized: more layers = more freedom to match detail, more chance of cheating realism.
- **`-bad_noise_layers`** (default `"17"`) — specific noise layers forced to zero because their high-frequency content tends to add artifacts. **This list is split on `.`** (e.g. `"3.5"`), not commas.
- **`-tile_latent`** — off by default. **Off**, the 18 style vectors are optimized independently (StyleGAN's expressive "W+" space). **On**, a single vector is tiled to all 18 layers (the tighter "W" space — more constrained, sometimes more stable).
- **`-opt_name` / `-learning_rate` / `-steps` / `-lr_schedule`** — standard gradient-descent controls: which optimizer (`adam` default; also `sgd`, `sgdm`, `adamax`), the step size, how many iterations, and how the learning rate ramps across them (`linear1cycledrop` warms up then decays, then drops at the end). On top of these, every parameter is projected back onto a fixed-radius sphere after each step — the paper's key "Riemannian" trick that keeps the search on the realistic manifold.
- **`-duplicates`** — because the problem is underdetermined (many faces downscale to the same blur), each run from a fresh random start finds a _different_ valid face. `-duplicates 3` gives you three to pick from; **`-seed`** instead fixes the random start for reproducibility.

For the full story, see the [PULSE paper](https://arxiv.org/abs/2003.03808) — especially the section on the latent-space search and the spherical prior — and the [StyleGAN paper](https://arxiv.org/abs/1812.04948) for what the latent and noise inputs (and the W / W+ spaces) actually are.

### `-steps` is the parameter that matters most (and the default is too low)

The default `-steps 100` is enough for an easy input and **badly** short for a hard one. Measured on an M4 Max, 2 seeds per cell, scoring the **downscaling L2 of the saved image** — PULSE's own objective, so lower is better:

| input          | 100 steps | 200                | 400     | 800                | GEOCROSS 100→800 |
| -------------- | --------- | ------------------ | ------- | ------------------ | ---------------- |
| `demo` (easy)  | 0.00143   | 0.00136            | 0.00136 | 0.00135 (−5%)      | 0.48 → 0.04      |
| `demo3` (hard) | 0.00402   | **0.00199 (−50%)** | 0.00200 | 0.00201            | 10.45 → 1.44     |
| `demo5` (hard) | 0.00672   | 0.00363            | 0.00222 | **0.00201 (−70%)** | 6.59 → 1.93      |

`demo3` **halves its error** just by running 200 steps instead of 100; `demo5` needs ~800. The easy input is already converged at 100 and gains nothing. Note that `GEOCROSS` (the realism prior) improves alongside `L2` — the extra steps are not buying pixel accuracy by wandering off the manifold; the faces get _more_ plausible, not less.

The flat plateau at `0.0020` is not a coincidence: `L2` is clamped at `-eps` (default `2e-3`), so once the run reaches the target it stops pushing. **Sitting on that floor is what convergence looks like.** When a run _doesn't_ get there, it now says so:

```
BEST (100) | L2: 0.0067 | GEOCROSS: 7.6394 | TOTAL: 1.0524 | time: 4.7 | it/s: 21.09
  NOT CONVERGED: best L2 0.00670 > eps 0.002 after 100 steps (still improving). Raise -steps (try 200).
```

If you see that line, the face you got back does **not** downscale to your input as closely as you asked — the search simply ran out of steps while still descending. Raise `-steps` until the warning goes away.

Two related findings, same measurements:

- **The `linear1cycledrop` schedule is right; don't switch to `fixed`.** At 400 steps on `demo5`, `fixed` is **65% worse** than the default. The mid-run learning-rate ramp is doing real work, even though it makes the loss non-monotone (which is why the best step is often not the last one).
- **A higher `-learning_rate` (0.8 vs the default 0.4) helps hard inputs** by roughly 8–10%, and costs nothing on easy ones.

### Performance notes (Apple Silicon)

Measured on an M4 Max, torch 2.12, batch 1. **Read the `-steps` section above first** — every optimization below, combined, is worth about **8%**, while raising `-steps` on a hard input is worth **50%**. The speed work matters because it makes more steps affordable, not because 8% was ever the prize.

What's already applied (no flags needed):

- StyleGAN's hand-rolled nearest-neighbour upsample (`view → expand → contiguous`) is replaced with `F.interpolate`, which is **bit-identical** in output and gradient and ~5% faster on MPS. The 2020-era idiom lost its edge; `F.interpolate` has since gotten better and nobody went back to re-check.
- The optimization loop no longer compares a GPU tensor against a Python float every step. That forced a host sync, drained the Metal command queue, and left the GPU idle waiting on Python (~3%).

Two levers that look obvious and **do not work** — don't spend an afternoon on them:

- **`-batch_size > 1` buys nothing.** Throughput is flat from batch 1 to 8 (20.9 → 21.7 img/s): a single 1024×1024 StyleGAN pass already saturates the GPU. Same goes for batching `-duplicates`.
- **`channels_last` is a _regression_ on MPS** (22.2 → 15.1 it/s, 0.68×). It's the reflex move on CUDA and it is wrong here.

If you're timing anything yourself: MPS dispatches asynchronously, so a wall-clock timer that doesn't sync first measures the _enqueue_ rate, not the compute — it will happily report impossible throughput. `device.py::sync_device()` exists for this.

### Half precision

`-precision mixed` runs the 256/512/1024 blocks (plus toRGB) in fp16 and leaves everything else — including the optimized latent, the noise, and the loss — in fp32. Adam's `eps=1e-8` is flat zero in fp16 and the L2 loss lives near `2e-3`; both are places where half precision quietly rots.

Why only those blocks? Because **cost and fp16-tolerance run in opposite directions**, which is the reverse of the obvious guess:

| block       | time saved in fp16    | max image error             |
| ----------- | --------------------- | --------------------------- |
| `1024×1024` | 2.44 ms (5.8%)        | **0.0060** (most tolerant)  |
| `512×512`   | 1.05 ms               | 0.0105                      |
| `64×64`     | **−0.26 ms (slower)** | 0.0261                      |
| `8×8`       | **−0.25 ms (slower)** | **0.0400** (least tolerant) |

The big late blocks hold all the compute and are the _cheapest_ to degrade — their error lands near the output and never propagates. The tiny early blocks have almost no compute, amplify their error through every downstream block, and are actually _slower_ in fp16 because the cast costs more than the math.

**The honest verdict:** mixed is ~13% faster per step but ~8% worse per step, so **at equal wall clock it is a wash** — it wins on step-starved inputs (spend the speedup on more `-steps`) and loses on ones that already converge. **The reliable win is memory: 2527 → 1535 MiB (−39%)**, which is what makes it worth having on a 16GB machine. `bf16` is deliberately not offered: same speed as fp16 on MPS, roughly 6× the error.

### Parameter sweep (worked example)

To make the knobs above concrete, here is the same input (`input/demo.png`) super-resolved five times. Every run uses `-seed 42 -steps 100` so the random starting latent is **identical** — the only thing changing is the one parameter noted, so any difference in the output is attributable to that parameter. Outputs are in [`readme_resources/experiments/`](readme_resources/experiments).

| Output              | Command (all share `-seed 42 -steps 100`)       | What's different                                              | Final `L2` | Final `GEOCROSS` | Converged |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------------- | ---------- | ---------------- | --------- |
| `baseline.png`      | `python run.py`                                 | defaults (`100*L2+0.05*GEOCROSS`, `noise_type trainable`, W+) | `0.0020`   | `0.523`          | step 100  |
| `tile_latent.png`   | `python run.py -tile_latent`                    | one shared latent (W space) instead of 18 (W+)                | `0.0020`   | `0.000`          | step 45   |
| `geocross_high.png` | `python run.py -loss_str "100*L2+1.0*GEOCROSS"` | 20× stronger realism/spread penalty                           | `0.0020`   | `0.030`          | step 100  |
| `no_geocross.png`   | `python run.py -loss_str "100*L2"`              | realism penalty removed entirely                              | `0.0020`   | _(n/a)_          | step 27   |
| `noise_fixed.png`   | `python run.py -noise_type fixed`               | noise frozen at random init, not optimized                    | `0.0020`   | `0.852`          | step 100  |

| baseline                                                 | tile_latent                                                    | geocross_high                                                      | no_geocross                                                    | noise_fixed                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------- |
| ![baseline](./readme_resources/experiments/baseline.png) | ![tile_latent](./readme_resources/experiments/tile_latent.png) | ![geocross_high](./readme_resources/experiments/geocross_high.png) | ![no_geocross](./readme_resources/experiments/no_geocross.png) | ![noise_fixed](./readme_resources/experiments/noise_fixed.png) |

Things to notice, tying back to the explanations above:

- **All five reach the same `L2` (`0.0020`).** Every variant is a genuinely valid solution — each downscales onto the input equally well. This is the underdetermined-problem point: the parameters don't change _whether_ it matches, they change _which_ realistic face you land on while matching.
- **`GEOCROSS` behaves exactly as described.** `tile_latent` drives it to `0.000` (with one shared latent there is no spread between the 18 vectors to penalize), `geocross_high` squeezes the 18 vectors tightly together (`0.030`), and `no_geocross` drops the term entirely.
- **Removing the realism term converges fastest** (`no_geocross` at step 27) because the optimizer only has to satisfy the pixel match — but that is also the configuration most free to drift off the realistic-face manifold on a harder input.
- Differences here are subtle because `demo.png` is an easy input that every setting solves comfortably; on lower-quality or more ambiguous inputs these knobs separate much more dramatically — which is exactly what the next example shows.

#### The same sweep on a harder input

`input/demo2.png` is the same face shrunk to a tiny **16×16**, so PULSE has to invent a 1024×1024 result from 1/4 as many pixels — a **64× upscale** (PULSE's headline case from the paper) instead of 32×. With so little to pin the answer down, the search has far more freedom, and the parameters now move the result much more.

| Output                    | What's different                  | Final `L2` | Final `GEOCROSS` | Converged |
| ------------------------- | --------------------------------- | ---------- | ---------------- | --------- |
| `demo2_baseline.png`      | defaults                          | `0.0020`   | `0.523`          | step 100  |
| `demo2_tile_latent.png`   | `-tile_latent` (W space)          | `0.0020`   | `0.000`          | step 28   |
| `demo2_geocross_high.png` | `-loss_str "100*L2+1.0*GEOCROSS"` | `0.0020`   | `0.026`          | step 100  |
| `demo2_no_geocross.png`   | `-loss_str "100*L2"`              | `0.0020`   | _(n/a)_          | step 20   |
| `demo2_noise_fixed.png`   | `-noise_type fixed`               | `0.0020`   | `0.572`          | step 100  |

| baseline                                                             | tile_latent                                                                | geocross_high                                                                  | no_geocross                                                                | noise_fixed                                                                |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| ![demo2 baseline](./readme_resources/experiments/demo2_baseline.png) | ![demo2 tile_latent](./readme_resources/experiments/demo2_tile_latent.png) | ![demo2 geocross_high](./readme_resources/experiments/demo2_geocross_high.png) | ![demo2 no_geocross](./readme_resources/experiments/demo2_no_geocross.png) | ![demo2 noise_fixed](./readme_resources/experiments/demo2_noise_fixed.png) |

Every variant still reaches the same `L2` (`0.0020`) — they all downscale onto the input equally well — but now they clearly disagree about _who the person is_. `baseline`, `geocross_high` and `noise_fixed` cluster around the same light-brown-haired, smiling man. `tile_latent` lands on a visibly different, darker-haired one. And `no_geocross` — the run with the realism term deleted — goes furthest: darkest hair, a warmer, redder skin tone, and, tellingly, **visible speckled colour artifacts** around the mouth and collar.

That last one is the argument for `GEOCROSS` in a single image. With nothing holding the 18 style vectors together, the optimizer is free to wander off the manifold of realistic faces, and it does — it finds a solution that downscales onto the input perfectly (`L2 0.0020`, and it gets there **fastest**, by step 20) while no longer looking entirely like a photograph. Matching the pixels and staying a plausible face are genuinely different objectives, and this is what it looks like when you only ask for the first one.

Same blurry input, same starting latent, different knobs → different believable people. That gap between "matches the pixels" and "which face you get" is the whole point of PULSE, and it widens as the input gets harder.
