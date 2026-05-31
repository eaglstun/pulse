# PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models

Code accompanying CVPR'20 paper of the same title. Paper: [arXiv:2003.03808](https://arxiv.org/abs/2003.03808) ([CVPR open-access PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Menon_PULSE_Self-Supervised_Photo_Upsampling_via_Latent_Space_Exploration_of_Generative_CVPR_2020_paper.pdf))

## NOTE

The original authors noted concern that PULSE would be used to identify individuals whose faces have been blurred out, and emphasized that this is impossible - **PULSE makes imaginary faces of people who do not exist, which should not be confused for real people.** It will **not** help identify or reconstruct the original image.

They also addressed concerns of bias in PULSE: **the [paper](https://arxiv.org/abs/2003.03808) includes a section, along with an accompanying model card, directly addressing this bias.**

---

## Fork notes

This fork updates the original CVPR'20 code so it runs on modern hardware and without the (now-defunct) hosted model weights:

- **Runs on CUDA, Apple Silicon (MPS/Metal), or CPU** — the device is auto-selected by `device.py`; no code changes needed. Force one with `PULSE_DEVICE=cpu` (or `mps`/`cuda`).
- **Loads model weights from local files** (`synthesis.pt`, `mapping.pt`, `shape_predictor_68_face_landmarks.dat`) when present in the repo root, falling back to download only if they're missing. The original Google Drive and self-hosted download links are no longer live, so place these files in the repo root yourself. `shape_predictor_68_face_landmarks.dat` is the stock dlib model (`http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`).
- `pulse.yml` is modernized to PyTorch 2.x (the same torch used for the MPS path), Python 3.13, and only the packages the code actually uses.

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

## What does it do?

Given a low-resolution input image, PULSE searches the outputs of a generative model (here, [StyleGAN](https://github.com/NVlabs/stylegan)) for high-resolution images that are perceptually realistic and downscale correctly.

![Transformation Preview](./readme_resources/transformation.gif)

## Usage

The main file of interest for applying PULSE is `run.py`. A full list of arguments with descriptions can be found in that file; the ones relevant to getting started are described below.

### Prereqs

You will need to install cmake first (required for dlib, which is used for face alignment). This fork runs on CUDA, Apple Silicon (MPS), or CPU (see [Fork notes](#fork-notes)); the original only supported CUDA and was tested on Linux and Windows. For the full set of required Python packages, create a Conda environment from the provided YAML, e.g.

```bash
conda env create -n pulse -f pulse.yml
conda activate pulse
```

You will also need the pretrained model weights in the repo root: `synthesis.pt`, `mapping.pt`, and `shape_predictor_68_face_landmarks.dat`. The code uses these local files directly if present. The original auto-download links (Google Drive, and a later self-hosted mirror) are no longer live, so obtain the files yourself and drop them in the repo root — `shape_predictor_68_face_landmarks.dat` is the stock dlib model from `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`. To restore auto-download, edit the URLs in `PULSE.py` and `align_face.py` to point at a host you control.

### Data

By default, input data for `run.py` should be placed in `./input/` (though this can be modified). However, this assumes faces have already been aligned and downscaled. If you have data that is not already in this form, place it in `realpics` and run `align_face.py` which will automatically do this for you. (Again, all directories can be changed by command line arguments if more convenient.) You will at this stage pic a downscaling factor.

Note that if your data begins at a low resolution already, downscaling it further will retain very little information. In this case, you may wish to bicubically upsample (usually, to 1024x1024) and allow `align_face.py` to downscale for you.

The dataset the original authors evaluated on was [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), but in their experience PULSE works with any picture of a realistic face.

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

| Argument                      | Default                | Description                                                                            |
| ----------------------------- | ---------------------- | -------------------------------------------------------------------------------------- |
| `-input_dir`                  | `input`                | input data directory                                                                   |
| `-output_dir`                 | `runs`                 | output data directory                                                                  |
| `-cache_dir`                  | `cache`                | cache directory for model weights                                                      |
| `-duplicates`                 | `1`                    | how many HR images to produce for every image in the input directory                   |
| `-batch_size`                 | `1`                    | batch size to use during optimization                                                  |
| `-seed`                       | _(none)_               | manual seed to use                                                                     |
| `-loss_str`                   | `100*L2+0.05*GEOCROSS` | loss function to use (weighted terms: `L2`, `L1`, `GEOCROSS`)                          |
| `-eps`                        | `2e-3`                 | target for the downscaling loss (L2); optimization stops contributing once within this |
| `-noise_type`                 | `trainable`            | `zero`, `fixed`, or `trainable`                                                        |
| `-num_trainable_noise_layers` | `5`                    | number of noise layers to optimize                                                     |
| `-tile_latent`                | `False` (flag)         | forcibly tile the same latent 18 times                                                 |
| `-bad_noise_layers`           | `17`                   | noise layers to zero out to improve image quality (split on `.`, e.g. `3.5`)           |
| `-opt_name`                   | `adam`                 | optimizer for projected gradient descent (`sgd`, `adam`, `sgdm`, `adamax`)             |
| `-learning_rate`              | `0.4`                  | learning rate to use during optimization                                               |
| `-steps`                      | `100`                  | number of optimization steps                                                           |
| `-lr_schedule`                | `linear1cycledrop`     | `fixed`, `linear1cycledrop`, or `linear1cycle`                                         |
| `-save_intermediate`          | `False` (flag)         | store and save intermediate HR and LR images during optimization                       |

**`align_face.py`** — (optional preprocessing) align + downscale raw photos:

| Argument       | Default    | Description                                                 |
| -------------- | ---------- | ----------------------------------------------------------- |
| `-input_dir`   | `realpics` | directory with unprocessed images                           |
| `-output_dir`  | `input`    | output directory                                            |
| `-output_size` | `32`       | size to downscale the input images to, must be a power of 2 |
| `-seed`        | _(none)_   | manual seed to use                                          |
| `-cache_dir`   | `cache`    | cache directory for model weights                           |
