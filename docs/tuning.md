# Tuning

← [back to the README](../README.md)

If you only read one page of these docs, make it this one.

Everything here was measured on an M4 Max with multiple seeds per cell, scored on the thing PULSE is actually trying to minimize: **the L2 distance between the saved image, shrunk back down, and your input.** Lower is better. The optimization is chaotic — a single run tells you almost nothing — so nothing below rests on one sample.

---

## The headline: `-steps` is too low, and it isn't close

The default is `-steps 100`. That's plenty for an easy photo and **badly** short for a hard one.

| input          | 100 steps | 200                | 400     | 800                | GEOCROSS 100 → 800 |
| -------------- | --------- | ------------------ | ------- | ------------------ | ------------------ |
| `demo` (easy)  | 0.00143   | 0.00136            | 0.00136 | 0.00135 (−5%)      | 0.48 → 0.04        |
| `demo3` (hard) | 0.00402   | **0.00199 (−50%)** | 0.00200 | 0.00201            | 10.45 → 1.44       |
| `demo5` (hard) | 0.00672   | 0.00363            | 0.00222 | **0.00201 (−70%)** | 6.59 → 1.93        |

`demo3` **halves its error** by running 200 steps instead of 100. `demo5` needs about 800. The easy input was already finished at 100 and gains nothing from more.

Put that next to the performance work. Every Apple-Silicon optimization in this repo — the faster upsample, the removed host sync, the compiled generator, half precision — is worth about **8% put together**. Changing one number is worth **50%**.

**That is the whole lesson.** The speed work matters because it makes more steps affordable. It was never the prize.

And this isn't a metric being gamed: notice that `GEOCROSS` — the realism term — improves _alongside_ `L2` (10.45 → 1.44). The extra steps are not buying pixel accuracy by wandering off the manifold of real faces. Both objectives get better together. The faces get **more** plausible, not less.

### What convergence actually looks like

Look at the plateau: `demo3` lands on 0.00199, then 0.00200, then 0.00201. Dead flat at `0.0020`.

That's not a coincidence, and it's not the optimizer giving up. **`L2` is clamped at `-eps`** (default `2e-3`). Once the match is close enough, that term stops pushing — by design, so the search doesn't spend its last hundred steps chasing pixel decimals at the cost of realism. So the loss can never go _below_ `eps`.

**Sitting on that floor is what convergence looks like.** Reaching `0.0020` means "done," not "stuck."

Which means the interesting question is what happens when a run _doesn't_ get there. It now says so:

```
BEST (100) | L2: 0.0067 | GEOCROSS: 7.6394 | TOTAL: 1.0524 | time: 4.7 | it/s: 21.09
  NOT CONVERGED: best L2 0.00670 > eps 0.002 after 100 steps (still improving). Raise -steps (try 200).
```

If you see that, the face you got back does **not** downscale onto your input as closely as you asked. The search was still descending when the step counter ran out. It isn't broken — it's unfinished. Raise `-steps` until the line goes away.

The default stays at 100, because raising it would silently double everyone's runtime and change everyone's outputs. You get told instead.

### Two related findings

- **Don't switch `-lr_schedule` to `fixed`.** At 400 steps on `demo5`, `fixed` is **65% worse** than the default `linear1cycledrop`. The mid-run learning-rate ramp looks wasteful — it deliberately _un-converges_ you halfway through — and it earns its keep anyway. (It's also why the loss is non-monotone, and why the best step is so often not the last one.)
- **`-learning_rate 0.8`** (vs. the default `0.4`) helps hard inputs by roughly 8–10% and costs nothing on easy ones. Worth trying; not made the default on the strength of two seeds.

---

## Worked example: one photo, five settings

Here is `input/demo.png` super-resolved five times. Every run shares `-seed 42 -steps 100`, so the random starting latent is **identical**. Exactly one parameter changes per row, which means any difference you see is attributable to that parameter and nothing else.

| Output          | Command (all share `-seed 42 -steps 100`) | What's different                         | `L2`     | `GEOCROSS` | Best step |
| --------------- | ----------------------------------------- | ---------------------------------------- | -------- | ---------- | --------- |
| `baseline`      | `python run.py`                           | the defaults                             | `0.0020` | `0.523`    | 100       |
| `tile_latent`   | `-tile_latent`                            | one shared latent (W) instead of 18 (W+) | `0.0020` | `0.000`    | 45        |
| `geocross_high` | `-loss_str "100*L2+1.0*GEOCROSS"`         | 20× stronger realism penalty             | `0.0020` | `0.030`    | 100       |
| `no_geocross`   | `-loss_str "100*L2"`                      | realism penalty deleted entirely         | `0.0020` | _(n/a)_    | 27        |
| `noise_fixed`   | `-noise_type fixed`                       | noise frozen at random init              | `0.0020` | `0.852`    | 100       |

| baseline                                                                            | tile_latent                                                                               | geocross_high                                                                                 | no_geocross                                                                               | noise_fixed                                                                               |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| <img src="../readme_resources/experiments/baseline.png" alt="baseline" width="160"> | <img src="../readme_resources/experiments/tile_latent.png" alt="tile_latent" width="160"> | <img src="../readme_resources/experiments/geocross_high.png" alt="geocross_high" width="160"> | <img src="../readme_resources/experiments/no_geocross.png" alt="no_geocross" width="160"> | <img src="../readme_resources/experiments/noise_fixed.png" alt="noise_fixed" width="160"> |

What to notice:

- **All five reach the same `L2`.** Every one of them is a genuinely valid answer — each shrinks back onto the input equally well. This is the underdetermined problem in a table: the parameters don't change _whether_ you match, they change _which_ face you land on while matching.
- **`GEOCROSS` behaves exactly as advertised.** `tile_latent` drives it to `0.000` (with one shared latent there's no spread between the 18 vectors to penalize). `geocross_high` squeezes them together (`0.030`). `no_geocross` deletes the term.
- **Deleting the realism term converges fastest** — `no_geocross` is done by step 27 — because the optimizer only has one job. Hold that thought.
- The differences here are subtle, because `demo.png` is an easy input that every setting solves comfortably. Which is exactly why the next example exists.

---

## The same sweep, on a harder input

`input/demo2.png` is the same face shrunk to **16×16** — a **64× upscale**, PULSE's headline case from the paper, from a quarter as many pixels. With that little to pin the answer down, the search has enormously more freedom, and the knobs start to matter.

| Output                | What's different                  | `L2`     | `GEOCROSS` | Best step |
| --------------------- | --------------------------------- | -------- | ---------- | --------- |
| `demo2_baseline`      | the defaults                      | `0.0020` | `0.523`    | 100       |
| `demo2_tile_latent`   | `-tile_latent` (W space)          | `0.0020` | `0.000`    | 28        |
| `demo2_geocross_high` | `-loss_str "100*L2+1.0*GEOCROSS"` | `0.0020` | `0.026`    | 100       |
| `demo2_no_geocross`   | `-loss_str "100*L2"`              | `0.0020` | _(n/a)_    | 20        |
| `demo2_noise_fixed`   | `-noise_type fixed`               | `0.0020` | `0.572`    | 100       |

| baseline                                                                                        | tile_latent                                                                                           | geocross_high                                                                                             | no_geocross                                                                                           | noise_fixed                                                                                           |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| <img src="../readme_resources/experiments/demo2_baseline.png" alt="demo2 baseline" width="160"> | <img src="../readme_resources/experiments/demo2_tile_latent.png" alt="demo2 tile_latent" width="160"> | <img src="../readme_resources/experiments/demo2_geocross_high.png" alt="demo2 geocross_high" width="160"> | <img src="../readme_resources/experiments/demo2_no_geocross.png" alt="demo2 no_geocross" width="160"> | <img src="../readme_resources/experiments/demo2_noise_fixed.png" alt="demo2 noise_fixed" width="160"> |

Every variant still hits the same `L2`. They all shrink onto the input equally well. But now they openly disagree about **who the person is.**

`baseline`, `geocross_high` and `noise_fixed` cluster around the same light-brown-haired, smiling man. `tile_latent` finds a visibly different, darker-haired one. Same blur, same starting latent, different knobs, different people — all of them correct.

### The argument for GEOCROSS in one image

And then there's `no_geocross`, which went furthest: darkest hair, a warmer and redder skin tone, and — look closely at the mouth and the collar — **visible speckled colour artifacts.**

That is what force 2 was holding back.

With nothing tying the 18 style vectors together, the optimizer is free to wander off the manifold of realistic faces. So it does. It finds a solution that downscales onto the input _perfectly_ (`L2 0.0020`) and gets there **faster than anything else** (step 20), and the result has stopped being entirely photographic.

Matching the pixels and staying a plausible human being are genuinely different objectives. That image is what it looks like when you only ask for the first one.

---

**Next:** [Apple Silicon notes](apple-silicon.md) — making the extra steps affordable · [How it works](how-it-works.md)
