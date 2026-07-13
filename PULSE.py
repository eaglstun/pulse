from stylegan import G_synthesis, G_mapping
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import os
import time
import torch
from loss import LossBuilder
from functools import partial
from drive import open_url
from device import device, sync_device


class PULSE(torch.nn.Module):
    # Load the frozen StyleGAN synthesis network and the cached gaussian_fit
    # (mean/std of the mapping network output), regenerating the latter from the
    # mapping network if it isn't already on disk.
    def __init__(self, cache_dir, verbose=True, compile_synthesis=False, precision='fp32'):
        super(PULSE, self).__init__()

        self.synthesis = G_synthesis().to(device)
        self.verbose = verbose
        self.precision = precision
        self.synth_dtype = {'fp32': torch.float32,
                            'fp16': torch.float16,
                            'mixed': torch.float32}[precision]

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print("Loading Synthesis Network")
        if Path("synthesis.pt").exists():
            self.synthesis.load_state_dict(torch.load("synthesis.pt", map_location=device))
        else:
            # with open_url("https://drive.google.com/uc?id=1TCViX1YpQyRsklTVYEJwdbmK91vklCo8", cache_dir=cache_dir, verbose=verbose) as f:
            with open_url("https://ericeaglstun.com/misc/synthesis.pt", cache_dir=cache_dir, verbose=verbose) as f:
                self.synthesis.load_state_dict(torch.load(f, map_location=device))

        for param in self.synthesis.parameters():
            param.requires_grad = False

        # Run the frozen generator in half precision, but keep the OPTIMISED variables
        # (latent, noise) and the loss in fp32 -- see forward(). Half-precision weights
        # cut conv bandwidth; half-precision Adam does not survive contact with reality
        # (its eps=1e-8 is flat zero in fp16, and the L2 loss lives around 2e-3).
        # bf16 is deliberately not offered: same speed as fp16 on MPS, ~6x the error.
        # Static loss scale for half-precision backward (see forward()). Measured: it does
        # NOT rescue whole-network fp16 (the damage is in the forward pass, not gradient
        # underflow), and scales above ~1024 overflow. Kept modest and only where half is
        # actually in play.
        self.grad_scale = float(os.environ.get("PULSE_GRAD_SCALE", 1024.0)) \
            if precision != 'fp32' else 1.0

        if precision == 'fp16':
            # Whole-network half. Fast but measurably worse output -- see -precision mixed.
            self.synthesis = self.synthesis.to(self.synth_dtype)
        elif precision == 'mixed':
            # Half only where it's free: the 256/512/1024 blocks are ~89% of the step time
            # but the most fp16-tolerant. See G_synthesis.set_high_res_precision.
            self.synthesis.set_high_res_precision(torch.float16, min_res=256)
        if self.verbose and precision != 'fp32':
            print(f"Synthesis Network precision: {precision} (grad scale {self.grad_scale:g})")

        # The generator is frozen and re-run with identical shapes every step, so it
        # is an ideal compile target: inductor fuses StyleGAN's long pointwise chains
        # (noise add -> lrelu -> instance norm -> style mod, 18 times per pass).
        # Warmup is ~1s, so it pays for itself within the first image. Degrade to eager
        # if the backend can't handle this torch/platform rather than failing the run.
        if compile_synthesis:
            try:
                self.synthesis = torch.compile(self.synthesis)
                if self.verbose:
                    print("Compiling Synthesis Network (first step will be slow)")
            except Exception as e:
                if self.verbose:
                    print(f"torch.compile unavailable, falling back to eager: {e}")

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        if Path("gaussian_fit.pt").exists():
            self.gaussian_fit = torch.load("gaussian_fit.pt", map_location=device)
        else:
            if self.verbose:
                print("\tLoading Mapping Network")
            mapping = G_mapping().to(device)

            if Path("mapping.pt").exists():
                mapping.load_state_dict(torch.load("mapping.pt", map_location=device))
            else:
                # with open_url("https://drive.google.com/uc?id=14R6iHGf5iuVx3DMNsACAl7eBr7Vdpd0k", cache_dir=cache_dir, verbose=verbose) as f:
                with open_url("https://ericeaglstun.com/misc/mapping.pt", cache_dir=cache_dir, verbose=verbose) as f:
                    mapping.load_state_dict(torch.load(f, map_location=device))

            if self.verbose:
                print("\tRunning Mapping Network")
            with torch.no_grad():
                torch.manual_seed(0)
                latent = torch.randn(
                    (1000000, 512), dtype=torch.float32, device=device)
                latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
                self.gaussian_fit = {"mean": latent_out.mean(
                    0), "std": latent_out.std(0)}
                torch.save(self.gaussian_fit, "gaussian_fit.pt")
                if self.verbose:
                    print("\tSaved \"gaussian_fit.pt\"")

    # Run projected gradient descent over the latent (and noise) for `steps`
    # iterations to find an HR face that downscales to `ref_im`. A generator that
    # yields (HR, LR) tensors: every step when save_intermediate, else just the best.
    def forward(self, ref_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate,
                **kwargs):

        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True

        # When running under DataParallel (CUDA) the input is already scattered
        # onto the GPU; otherwise (MPS/CPU) move it onto the target device here.
        ref_im = ref_im.to(device)

        batch_size = ref_im.shape[0]

        # Generate latent tensor
        if(tile_latent):
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device=device)
        else:
            latent = torch.randn(
                (batch_size, 18, 512), dtype=torch.float, requires_grad=True, device=device)

        # Generate list of noise tensors
        noise = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on

        for i in range(18):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2**(i//2+2), 2**(i//2+2))

            if(noise_type == 'zero' or i in [int(layer) for layer in bad_noise_layers.split('.')]):
                new_noise = torch.zeros(res, dtype=torch.float, device=device)
                new_noise.requires_grad = False
            elif(noise_type == 'fixed'):
                new_noise = torch.randn(res, dtype=torch.float, device=device)
                new_noise.requires_grad = False
            elif (noise_type == 'trainable'):
                new_noise = torch.randn(res, dtype=torch.float, device=device)
                if (i < num_trainable_noise_layers):
                    new_noise.requires_grad = True
                    noise_vars.append(new_noise)
                else:
                    new_noise.requires_grad = False
            else:
                raise Exception("unknown noise type")

            noise.append(new_noise)

        var_list = [latent]+noise_vars

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, eps).to(device)

        # Track the best iterate on-device. Comparing a GPU tensor against a Python
        # float (`if loss < min_loss`) forces a host sync every single step, which
        # drains the Metal command queue and leaves the GPU idle waiting on Python.
        # Keeping these as tensors and selecting with torch.where lets the queue stay
        # deep; the values are read back once, after the loop.
        min_loss = torch.full((), float('inf'), device=device)
        min_l2 = torch.full((), float('inf'), device=device)
        loss_history = []
        best_im = None
        start_t = time.time()
        gen_im = None

        if self.verbose:
            print("Optimizing")
        for j in range(steps):
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, 18, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(
                latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Cast into the generator's dtype at its boundary and come straight back out
            # to fp32. The optimised variables stay fp32 leaves, so Adam and the spherical
            # projection keep full precision and only the (frozen, bandwidth-bound) conv
            # stack runs in half. Autograd puts the matching casts in the backward pass.
            if self.synth_dtype != torch.float32:
                syn_out = self.synthesis(latent_in.to(self.synth_dtype),
                                         [n.to(self.synth_dtype) for n in noise])
                syn_out = syn_out.float()
            else:
                syn_out = self.synthesis(latent_in, noise)

            # Normalize image to [0,1] instead of [-1,1]
            gen_im = (syn_out+1)/2

            # Calculate Losses (in fp32: eps is 2e-3 and L2 lands near it, which is
            # exactly the range where fp16 rounding starts eating the signal)
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            # Track the best-so-far image without ever touching the host. `improved` is
            # a 0-dim bool tensor, so torch.where picks the new image or keeps the old
            # one entirely on the GPU. Stash the per-step losses (still on device) and
            # format the log after the loop, where one sync is free.
            improved = loss < min_loss
            min_loss = torch.minimum(min_loss, loss.detach())
            min_l2 = torch.minimum(min_l2, torch.as_tensor(
                loss_dict['L2'], device=device).detach())

            if best_im is None:
                best_im = gen_im.detach().clone()
            else:
                best_im = torch.where(improved, gen_im.detach(), best_im)

            loss_history.append({k: torch.as_tensor(v, device=device).detach()
                                 for k, v in loss_dict.items()})

            # Save intermediate HR and LR images
            if(save_intermediate):
                yield (best_im.cpu().detach().clamp(0, 1), loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            # Loss scaling for fp16. Gradients flowing back through the half-precision
            # generator can underflow fp16's ~6e-8 floor and flush to zero, which shows
            # up not as a crash but as a quietly worse optimum. Scale the loss up before
            # backward, then unscale the (fp32) grads before the optimiser sees them, so
            # the step is mathematically unchanged.
            if self.grad_scale != 1.0:
                (loss*self.grad_scale).backward()
                for p in var_list:
                    if p.grad is not None:
                        p.grad.div_(self.grad_scale)
            else:
                loss.backward()
            opt.step()
            scheduler.step()

        # The loop above never syncs, so Python has been running ahead of the GPU.
        # Wait for the queued work to actually finish before stopping the clock,
        # otherwise the reported it/s is the enqueue rate and wildly overstated.
        sync_device()

        # The single host sync: everything above stayed on-device, so read back once
        # here to find which step won and format its losses.
        best_summary = ""
        if loss_history:
            totals = torch.stack([h['TOTAL'] for h in loss_history])
            best_j = int(totals.argmin())
            best_summary = f'BEST ({best_j+1}) | '+' | '.join(
                [f'{x}: {float(y):.4f}' for x, y in loss_history[best_j].items()])

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose:
            print(best_summary+current_info)

        # Yield the BEST iterate, not the last one. This used to hand back gen_im (the
        # final step) as HR while taking LR from best_im -- two different images, and
        # neither matched the "BEST (n)" the line above advertises. The loss is not
        # monotone (the 1cycle schedule raises the LR mid-run, and GEOCROSS trades off
        # against L2), so on some inputs the last step is genuinely worse than the best.
        yield (best_im.cpu().detach().clamp(0, 1), loss_builder.D(best_im).cpu().detach().clamp(0, 1))
        
        # else:
        #     print("Could not find a face that downscales correctly within epsilon")
        #     print("min_l2", min_l2)
        #     print("eps", eps)
