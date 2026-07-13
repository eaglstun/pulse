# How it works

← [back to the README](../README.md)

Most super-resolution works the obvious way: train a network on millions of (blurry, sharp) pairs until it learns to map one to the other.

PULSE doesn't do that, and the sleight of hand is the whole reason it's interesting.

**It never trains anything.** It takes a StyleGAN generator that is already trained — a frozen machine that turns about 512 numbers into a photorealistic face — and adjusts _those input numbers_ by gradient descent, leaving the machine itself untouched. Normal training does the reverse: it tweaks the machine and leaves the inputs alone. PULSE flips it. No paired data, no training run. Just search.

```text
       normal training:   fix the inputs,  move the network weights
                  PULSE:   fix the network, move the inputs
```

That's the trick. Everything below is a consequence of it.

### Why bother

Here's the paper's own figure. Left to right: the low-res input, then bicubic, FSRNET, and FSRGAN — all of them trained the conventional way — then PULSE. Look at the crop along the bottom row, and then look at the far right.

![Comparison from the PULSE paper: a low-res input, then bicubic, FSRNET, FSRGAN and PULSE reconstructions, with a zoomed crop of the hair beneath each. On the right, the PULSE output is downscaled back down and lands on the original input.](../readme_resources/014.jpeg)

The conventional methods hedge. Asked to produce detail they cannot know, they average over every plausible answer, and the average of many sharp things is a blurry thing. That's why the hair in the crop is mush — it's the mean of every hairstyle consistent with those pixels.

PULSE refuses to average. It commits to _one_ specific face from the space of realistic ones. The hair has individual strands because the face it picked has individual strands. It is not more _accurate_ — it may well be the wrong person entirely — but it is a real answer instead of the smeared centroid of a thousand answers.

And the arrow on the right is the whole contract: shrink PULSE's output back down, and you land on the input you started from.

![Another comparison from the PULSE paper, same layout.](../readme_resources/034.jpeg)

---

## What "latent space" means

A generator is a function that turns a short list of numbers into an image:

```text
latent vector  →  StyleGAN  →  image
 (512 numbers)                 (1024×1024 face)
```

The **latent space** is the space of all possible input vectors. Every point in it maps to some face. It's "latent" — hidden — because those numbers aren't pixels; they're an abstract code the network learned to expand into a picture.

Think of a synthesizer. You don't draw the sound wave; you turn a few dozen knobs and the synth produces audio. The space of all knob settings is its latent space. StyleGAN is that idea for faces: hand it ~512 knob values, get a photorealistic person; nudge them, and the person changes smoothly — older, smiling, different hair.

What makes this _useful_ is that a trained GAN's latent space is **organized**, not random:

- **Nearby points make similar images**, so you can slide toward a target with gradient descent instead of guessing.
- **Directions carry meaning** — one direction adds a smile, another rotates the head, another ages the face.
- **Realistic faces live on a thin surface** (a _manifold_) inside the space. Points on it look like people. Points far off it look like static.

PULSE exploits all three at once. It searches for an input whose output downscales onto your photo (using the smoothness), while forcing the search to stay on the realistic-face surface (so you get a person, not static that happens to shrink correctly).

## The two forces

Every knob in this repo is a knob on a tug-of-war between two things:

1. **"Downscale correctly."** Shrink the generated face; it must land on your input. → the `L2`/`L1` loss terms.
2. **"Stay a real face."** The latent must stay in the region StyleGAN considers natural. → the `GEOCROSS` term, plus a geometric constraint.

Force 1 alone will happily hand you something that matches your pixels and is not quite a photograph. There's [a picture of exactly that](tuning.md#the-argument-for-geocross-in-one-image) in the tuning docs — it's the most instructive image in the repo.

Force 2 alone gives you a beautiful face that looks nothing like your input.

The interesting behavior all lives in the middle.

### The spherical trick

There's a third thing, and it's the paper's actual contribution. After every gradient step, each parameter is **projected back onto a sphere of its original radius**. The search is allowed to move _around_ the surface but never to drift toward the origin or off to infinity.

Why: in a high-dimensional Gaussian, essentially all the probability mass sits in a thin shell at a particular radius — not near the center, where your intuition says it should be. So "realistic latents" are approximately "latents on that shell." Constraining the search to the sphere keeps it in the region StyleGAN was actually trained on. It's a cheap, geometric way to say _stay plausible_, and it's why the results look like people instead of the smeared nightmares you get from naive latent optimization.

Implementation lives in `SphericalOptimizer.py`, and it's about ten lines. Good ideas often are.

---

## What every knob actually does

The [CLI reference](getting-started.md#command-line-reference) says _what_ each flag is. This says _why_ it exists.

- **`-loss_str`** — the objective, written as `weight*TERM+weight*TERM`. The default `100*L2+0.05*GEOCROSS` means "mostly match the pixels, with a light realism nudge."
  - **`L2` / `L1`** — distance between the **downscaled** output and your input. This is force 1. `L1` shrugs off outlier pixels more than `L2` does.
  - **`GEOCROSS`** — a _geodesic cross_ penalty. The latent is really 18 separate 512-dimensional style vectors, one per StyleGAN layer. `GEOCROSS` measures how far apart they've spread on the sphere and penalizes the spread. Keeping them together ties the result to the kind of faces StyleGAN naturally makes. Loosen it and you get more per-layer detail — and more chance of artifacts.
- **`-eps`** (default `2e-3`) — a "good enough" floor. Once the downscaling match is within `eps`, that term **stops pushing**, so the optimizer doesn't spend its remaining steps chasing the last decimal place of pixel accuracy at the expense of realism. This matters more than it sounds: it means the loss _bottoms out_ at `eps`, and [sitting on that floor is what convergence looks like](tuning.md#what-convergence-actually-looks-like).
- **`-noise_type`** + **`-num_trainable_noise_layers`** — StyleGAN injects tiny per-pixel noise at each layer to render fine texture: pores, stray hairs, skin grain. `zero` ignores it. `fixed` rolls a random texture and freezes it. `trainable` optimizes it too, which gives the search more freedom to match detail — and more rope to cheat with.
- **`-bad_noise_layers`** (default `17`) — noise layers forced to zero because their high-frequency content tends to add artifacts. **This list splits on `.`** — `"3.5"` means layers 3 and 5 — not commas. A genuine trap; it will silently do nothing if you use commas.
- **`-tile_latent`** — off by default. **Off**, the 18 style vectors are optimized independently (StyleGAN's expressive **W+** space). **On**, one vector is tiled to all 18 layers (the tighter **W** space — more constrained, sometimes more stable, always less expressive).
- **`-duplicates`** — the problem is underdetermined: many faces shrink to the same blur. Each run from a fresh random start finds a _different_ valid one. `-duplicates 3` gives you three people to choose from. **`-seed`** does the opposite: it fixes the random start so a run is reproducible.
- **`-steps` / `-learning_rate` / `-lr_schedule` / `-opt_name`** — ordinary gradient-descent controls, except `-steps` is not ordinary at all in this repo. **It is the highest-leverage parameter in the system and the default is too low.** [Go read why](tuning.md).

For the real derivation, the [PULSE paper](https://arxiv.org/abs/2003.03808) is short and readable (there's a Markdown transcription in [`paper.md`](../paper.md)), and the [StyleGAN paper](https://arxiv.org/abs/1812.04948) explains the latent and noise inputs and the W / W+ distinction.

---

## The code

The optimization loop is the whole system. It's in `PULSE.py::forward()`, and it's short:

1. **The latent and the noise are the variables.** The generator is frozen (`requires_grad = False`). A `(batch, 18, 512)` latent plus up to `num_trainable_noise_layers` noise tensors are what gets optimized.
2. **Re-parameterize the latent** so that optimizing a plain standard-normal variable lands you in the distribution StyleGAN's mapping network actually produces.
3. **Generate, downscale, compare.** The downscaler (`bicubic.py`) is a fixed bicubic convolution — and crucially it is **differentiable**, so gradients flow back through the shrinking step. That is the hinge the entire method turns on.
4. **Step, then project back onto the sphere** (`SphericalOptimizer.py`).
5. **Repeat.** Keep the best iterate seen.

| file                    | what it is                                             |
| ----------------------- | ------------------------------------------------------ |
| `PULSE.py`              | the optimization loop — the core                       |
| `stylegan.py`           | the frozen generator                                   |
| `loss.py`               | `LossBuilder` — parses `-loss_str`, computes the terms |
| `bicubic.py`            | the differentiable downscaler                          |
| `SphericalOptimizer.py` | the radius-preserving projection                       |
| `run.py`                | CLI, dataset, writing PNGs                             |
| `align_face.py`         | dlib face alignment (preprocessing)                    |
| `device.py`             | picks CUDA / MPS / CPU                                 |

### The bug that lived for six years

Worth telling, because it's a good lesson about generators.

The loop tracks the **best** iterate it has seen — `best_im` — because the loss is _not_ monotone. The learning-rate schedule deliberately ramps back up mid-run, so the last step is frequently worse than something you passed 40 steps ago.

At the end, the original code did this:

```python
yield (gen_im.clone()..., loss_builder.D(best_im)...)
#      ^^^^^^ the LAST image        ^^^^^^^ the BEST image
```

Two different images. The high-res one you keep came from the final step; the low-res preview beside it came from the best step. Neither matched the `BEST (n)` the program printed one line above.

`git log` tells the story: before a June 2020 commit that converted `forward()` into a generator, the function ended with `return best_im`. Converting it to `yield` silently swapped in `gen_im` for the high-res half and nobody noticed, because on easy inputs the best step usually _is_ the last one — so the bug is invisible exactly when you're testing with easy inputs.

Fixed. It now yields `best_im` for both, which is what the pre-2020 code did and what the printed log claims.

---

**Next:** [Tuning](tuning.md) — the parameters that actually change your output · [Apple Silicon notes](apple-silicon.md)
