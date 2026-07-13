# Apple Silicon notes

← [back to the README](../README.md)

Everything here is measured on an M4 Max, torch 2.12, batch 1, on an idle machine.

**Read [Tuning](tuning.md) first.** Every optimization on this page, added together, is worth about **8%**. Raising `-steps` on a hard input is worth **50%**. The speed work matters because it buys you steps — not because 8% was ever the point. Keep that ordering in your head or you'll spend a week shaving milliseconds off a run that was going to hand you an unfinished face anyway.

With that said, the 8% is free, and the details are genuinely interesting.

---

## The torch 2.13 trap

**`torch` is pinned below 2.13 on purpose. Don't lift it.**

torch 2.13.0 regressed the MPS **backward** pass by ~25% on this model: **22.4 it/s on torch 2.10–2.12, 16.7 on 2.13.** The forward pass is completely unaffected — actually marginally _faster_ — so nothing looks wrong. You just quietly lose a quarter of your throughput, and you will never notice unless you profile.

The cause is a nice little lesson in GPU kernel design. torch 2.13 moved `sum` off Apple's MPSGraph onto a hand-written Metal kernel that assigns **one 32-lane SIMD group per output element.** That's a fine choice when you're reducing a big dimension down to a small one. It falls apart when the dimension you're reducing is _tiny_ — you burn 30 idle lanes and a full 32-lane shuffle to add two numbers, across millions of threadgroups.

And "reduce a tiny dimension" is precisely the shape produced by **the backward pass of an `expand` or a broadcast.** Which is to say: every upsample-by-expand, every per-channel bias, every FiLM/AdaIN-style conditioning layer. In this model, `expand().contiguous()` backward went **36× slower.**

Hold the tensor size constant and vary _only_ the size of the reduced dimension, and the shape of the bug is unmistakable:

| reduced extent | torch 2.12 | torch 2.13          |
| -------------- | ---------- | ------------------- |
| 2              | 0.93 ms    | **25.63 ms**        |
| 8              | 0.29 ms    | 6.42 ms             |
| 32             | 0.26 ms    | 1.64 ms             |
| 1024           | 0.25 ms    | **0.24 ms** (fine!) |

2.12 is flat. 2.13 scales as `1/extent`. At extent 1024 the new kernel is actually _better_ than the old one — it isn't bad code, it's code optimized for the wrong regime.

This has been [root-caused and fixed upstream-style](https://github.com/pytorch/pytorch) (the patch restores full parity, 22.3 it/s), but **the fix is not in any released wheel yet.** So the cap stays. Lift it when a PyTorch release actually contains the fix — and re-benchmark before you do.

If you must use 2.13 for some other reason: `-compile` sidesteps the whole thing, because the compiler fuses the offending reductions away.

---

## What's already applied

No flags needed; these are just on.

- **The upsample.** StyleGAN's hand-rolled nearest-neighbour upsample (`view → expand → contiguous`) is replaced with `F.interpolate`. It is **bit-identical** — same output, same gradients — and ~5% faster on MPS. The 2020 idiom was the right call in 2020; `F.interpolate` has since gotten better and nobody went back to re-check. Worth remembering as a general suspicion about ported code of that vintage.
- **The host sync.** The optimization loop used to compare a GPU tensor against a Python float every single step (`if loss < min_loss`). That forces a GPU→CPU synchronization, which drains the Metal command queue and leaves the GPU sitting idle waiting on Python. Now everything stays on-device and is read back once, at the end. Worth ~3%.

## Two levers that do not work

Both of these are the obvious thing to try. Both are wrong here. Don't spend an afternoon on them.

- **`-batch_size > 1` buys nothing.** Throughput is flat from batch 1 to batch 8 (20.9 → 21.7 img/s). One 1024×1024 StyleGAN pass already saturates the GPU; there is no idle capacity for a second image to use. Batching `-duplicates` doesn't help either.
- **`channels_last` is a _regression_** — 22.2 → 15.1 it/s, a 0.68× slowdown. It's the reflex move on CUDA and it is simply wrong on Metal.

## `torch.compile`

`-compile` is **opt-in, and deliberately not the default.**

It's about **11% faster per step** (40 ms vs 44.5 ms). But it costs a one-off warmup — roughly 1 second with a warm inductor cache, ~7 seconds cold — which makes it a net **loss** at the default 100 steps (19.3 vs 20.9 it/s all-in). It starts winning past about 200 steps, or across many images in one process.

Which, given [what the tuning docs say about step counts](tuning.md), is a happier coincidence than it sounds: the runs that need `-compile` are exactly the runs that need more steps.

## Half precision

`-precision mixed` runs the 256/512/1024 blocks (and toRGB) in fp16, leaving everything else in fp32 — including the optimized latent, the noise, and the loss. That split is not arbitrary: Adam's `eps=1e-8` is flat zero in fp16, and the L2 loss lives right around `2e-3`. Both are places where half precision quietly rots the answer without ever crashing.

**Why only those blocks?** Because cost and fp16-tolerance run in **opposite directions**, which is the reverse of what you'd guess:

| block       | time saved in fp16     | max image error             |
| ----------- | ---------------------- | --------------------------- |
| `1024×1024` | 2.44 ms (5.8%)         | **0.0060** — most tolerant  |
| `512×512`   | 1.05 ms                | 0.0105                      |
| `64×64`     | **−0.26 ms (slower!)** | 0.0261                      |
| `8×8`       | **−0.25 ms (slower!)** | **0.0400** — least tolerant |

The big late blocks hold essentially all the compute _and_ are the cheapest to degrade — their error lands near the output and never propagates. The tiny early blocks have almost no compute, amplify their error through every downstream block, and are actually **slower** in fp16 because the cast costs more than the arithmetic it saves.

So: fp16 pays only at the top of the stack. (One more wrinkle: those blocks are contiguous, so the cast happens _once_ on entry and once on exit. Casting at every block boundary round-trips a 1024×1024 activation four times and gives back half the speedup — for zero numerical difference.)

**The honest verdict:** mixed is ~13% faster per step but ~8% worse per step. At **equal wall clock it's a wash** — it wins on step-starved inputs, where you can spend the speedup on more `-steps`, and loses on inputs that already converge. Take it for the memory, not the speed: **2527 → 1535 MiB, a 39% cut**, which is the difference between running and not on a 16GB machine.

`bf16` is deliberately not offered. Same speed as fp16 on MPS, roughly **6× the error**. There's no version of that trade worth making.

---

## If you benchmark this yourself

**MPS dispatches asynchronously.** Python enqueues GPU work and returns immediately. A wall-clock timer that doesn't synchronize first is measuring the _enqueue rate_, not the computation — and it will cheerfully report throughput that is physically impossible. (During this work it briefly claimed 72 it/s, which is roughly 3× the hardware.)

`device.py::sync_device()` exists for exactly this. Call it before you stop the clock.

Two more things that will lie to you:

- **Other GPU work.** Anything else touching Metal — another training run, a test suite, even Photos' media analysis — silently wrecks the numbers. A contended run here read 12–16 it/s with 12% spread where a clean one read 19.5 with a standard deviation of 0.07. Check the machine is idle first.
- **The chaotic optimization.** Two runs with different seeds land on different faces with different losses. Any quality claim needs several seeds, or it's noise with a confident voice.

---

**Next:** [Tuning](tuning.md) — where the real wins are · [How it works](how-it-works.md)
