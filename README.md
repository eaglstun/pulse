# PULSE — revived

Give it a blurry 16×16 face. It hands back a sharp 1024×1024 one.

![The search running: a pixelated input image resolving into a sharp, photorealistic face, with an inset showing the result being downscaled back onto the original input.](./readme_resources/transformation.gif)

That's not a filter being applied. That's a search — a hundred-odd steps of gradient descent hunting through the space of every face a GAN can draw, looking for one that shrinks back down onto your blur. The inset in the corner is the receipt.

Which leads to the part people find unsettling.

It does **not** un-blur your photo. It invents a person who happens to shrink down to your photo. Run it twice and get two different people, both correct:

<img src="./readme_resources/experiments/demo2_baseline.png" alt="A face PULSE invented from a 16x16 blur" width="180"> <img src="./readme_resources/experiments/demo2_tile_latent.png" alt="A different face PULSE invented from the same blur" width="180"> <img src="./readme_resources/experiments/demo2_no_geocross.png" alt="A third face PULSE invented from the same blur" width="180">

_Three strangers. One 16×16 input. All three shrink back down onto it exactly._

That isn't a disclaimer bolted onto the front of this README. It's the mechanism, and it's the reason the thing is worth reviving at all.

---

## ⚠️ Read this part

The original authors were blunt about this and so am I: **PULSE makes imaginary faces of people who do not exist.** It **cannot** un-blur a redacted face and identify someone. The detail the blur destroyed is gone; nothing brings it back. What comes back is a plausible invention.

The search also leans toward the kinds of faces StyleGAN saw most of in training. The [paper](https://arxiv.org/abs/2003.03808) has a section and a model card addressing that bias directly. Read them before you do anything with this beyond making pictures.

> _Any resemblance to actual persons, living or dead, is purely coincidental — and also, mathematically, the entire point._

---

## What this fork is

The original PULSE is from 2020, and 2020 is a foreign country. The code assumed an NVIDIA GPU. The conda environment pinned Python 3.8 and PyTorch 1.5 to exact six-year-old builds. The model weights lived on Google Drive links that have since gone dark.

This fork drags it into the present:

- **Runs on Apple Silicon (MPS), CUDA, or plain CPU** — auto-detected, no code changes.
- **Modern stack** — Python 3.13, PyTorch 2.x, and only the packages the code actually imports.
- **Two real bugs fixed**, including one that had been quietly saving the wrong image since 2020 (details in [How it works](docs/how-it-works.md#the-bug-that-lived-for-six-years)).
- **Honest about when it fails.** The default settings under-converge hard inputs, and it now tells you so instead of handing back an unfinished face. This turned out to matter more than every speed optimization combined — see [Tuning](docs/tuning.md).

There's a long-form writeup of the revival — dead links, CUDA-only assumptions, a silent failure mode, and an interactive playground with 240 real reconstructions — at **[Necromancy for Neural Nets](https://ai.ericeaglstun.com/deep-dives/reviving-pulse-apple-silicon/)**.

## Quickstart

```bash
uv venv --python 3.13 && uv pip install -r requirements.txt
source .venv/bin/activate

# put synthesis.pt + shape_predictor_68_face_landmarks.dat in the repo root first --
# the original download links are dead. See docs/getting-started.md

python run.py                 # every *.png in input/  ->  runs/
```

If it prints `NOT CONVERGED`, it means what it says: the search ran out of road. Raise `-steps`.

Full setup, including where to get the weights: **[Getting started](docs/getting-started.md)**.

## Where to go next

|                                                  |                                                                                                                                                                                           |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[Getting started](docs/getting-started.md)**   | Install, model weights, preparing photos, the full CLI reference.                                                                                                                         |
| **[How it works](docs/how-it-works.md)**         | Latent space, the two forces pulling on the search, and what every knob actually does. Start here if you want to _understand_ it rather than run it.                                      |
| **[Tuning](docs/tuning.md)**                     | Worked parameter sweeps with real images, and the single most important finding in this repo: **the default `-steps` is too low, and fixing that beats every optimization put together.** |
| **[Apple Silicon notes](docs/apple-silicon.md)** | Performance on an M-series Mac. What helped, what didn't, and the PyTorch version that will quietly cost you 25% if you let it.                                                           |
| **[`paper.md`](paper.md)**                       | The CVPR'20 paper, transcribed to Markdown.                                                                                                                                               |

## Credit

PULSE is by Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin — [CVPR'20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.pdf), [arXiv:2003.03808](https://arxiv.org/abs/2003.03808). The generator is [StyleGAN](https://github.com/NVlabs/stylegan) (NVIDIA), by way of the [lernapparat](https://github.com/lernapparat/lernapparat) PyTorch port.

This fork is maintenance and measurement. The beautiful, weird idea is theirs.
