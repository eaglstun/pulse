#Modified from https://github.com/lernapparat/lernapparat/

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

import numpy as np


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""
    
    # Initialize the weight with He scaling, optionally deferring the scale to
    # runtime (equalized learning rate) and applying a learning-rate multiplier.
    def __init__(self, input_size, output_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size**(-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    # Apply the linear layer, scaling weight and bias by their runtime multipliers.
    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    # Initialize the conv weight with He scaling (optionally equalized-LR), and
    # set up optional upscaling and an optional intermediate (e.g. blur) op.
    def __init__(self, input_channels, output_channels, kernel_size, gain=2**(0.5), use_wscale=False, lrmul=1, bias=True,
                 intermediate=None, upscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size **
                         2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(
            output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    # Apply the conv, using the fused upscale+conv path at high resolutions and
    # otherwise an optional separate upscale, intermediate op, and bias add.
    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + \
                w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(
                x, w, stride=2, padding=int((w.size(-1)-1)//2))
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=int(self.kernel_size//2))
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None,
                         padding=int(self.kernel_size//2))

        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    # Per-channel noise weights, zero-initialized; self.noise can be set to inject
    # pre-defined noise instead of random noise.
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    # Add per-pixel noise (generated, supplied, or pre-set) scaled per channel.
    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(
                2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noiselayers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    # A linear layer mapping the latent to a per-channel scale and bias (AdaIN).
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)

    # Modulate x with the style: x * (scale + 1) + bias, derived from the latent.
    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class PixelNormLayer(nn.Module):
    # Store the epsilon used to avoid division by zero during normalization.
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    # Normalize each pixel's feature vector to unit length across channels.
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    # Build a separable low-pass (blur) kernel from the 1-D `kernel` and register
    # it as a buffer; optionally normalized to sum to 1 and/or flipped.
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    # Depthwise-convolve x with the blur kernel (one kernel copy per channel).
    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2)-1)/2),
            groups=x.size(1)
        )
        return x


# Nearest-neighbor upscale a NCHW tensor by an integer factor (optionally gained).
#
# The original hand-rolled this as view/expand/contiguous. F.interpolate is
# bit-identical here (output and gradient) and faster on MPS: the expand's backward
# is a sum-reduction over the expanded dims, which Metal handles poorly. Measured on
# M4 Max: +6.7% end-to-end on torch 2.12. On torch 2.13 the expand backward degrades
# 36x (see pulse.yml), so this also keeps us off that cliff.
def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        x = F.interpolate(x, scale_factor=factor, mode='nearest')
    return x


class Upscale2d(nn.Module):
    # Store the integer upscale factor and gain for use in forward.
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    # Module wrapper around the upscale2d function.
    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):
    # Build the mapping network: pixel-norm followed by 8 fully-connected layers
    # with the chosen nonlinearity, mapping Z (512) to W (512).
    def __init__(self, nonlinearity='lrelu', use_wscale=True):
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        layers = [
            ('pixel_norm', PixelNormLayer()),
            ('dense0', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense0_act', act),
            ('dense1', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense1_act', act),
            ('dense2', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense2_act', act),
            ('dense3', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense3_act', act),
            ('dense4', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense4_act', act),
            ('dense5', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense5_act', act),
            ('dense6', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense6_act', act),
            ('dense7', MyLinear(512, 512, gain=gain,
                                lrmul=0.01, use_wscale=use_wscale)),
            ('dense7_act', act)
        ]
        super().__init__(OrderedDict(layers))

    # Run the input latent through the sequential mapping layers.
    def forward(self, x):
        x = super().forward(x)
        return x


class Truncation(nn.Module):
    # Store the average latent (as a buffer) and the truncation strength/extent.
    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer('avg_latent', avg_latent)

    # Interpolate the first max_layer latents toward the average by `threshold`
    # (the truncation trick), leaving later layers unchanged.
    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return torch.where(do_trunc, interp, x)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    # Assemble the per-layer epilogue: optional noise add, activation, optional
    # pixel-/instance-norm, and an optional style modulation from the latent.
    def __init__(self, channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        layers = []
        if use_noise:
            self.noise = NoiseLayer(channels)
        else:
            self.noise = None
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNormLayer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(
                dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    # Apply noise, activation/norm stack, then style modulation in sequence.
    def forward(self, x, dlatents_in_slice=None, noise_in_slice=None):
        if(self.noise is not None):
            x = self.noise(x, noise=noise_in_slice)
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class InputBlock(nn.Module):
    # The first synthesis block (4x4): a learned constant (or dense-from-latent)
    # input followed by two epilogues with a 3x3 conv between them.
    def __init__(self, nf, dlatent_size, const_input_layer, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            # tweak gain to match the official implementation of Progressing GAN
            self.dense = MyLinear(dlatent_size, nf*16,
                                  gain=gain/4, use_wscale=use_wscale)
        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    # Produce the 4x4 feature map from the constant/dense input and two epilogues.
    def forward(self, dlatents_in_range, noise_in_range):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(
                batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0], noise_in_range[0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise_in_range[1])
        return x


class GSynthesisBlock(nn.Module):
    # A higher-resolution synthesis block: upsampling 3x3 conv (with optional blur)
    # plus two epilogues and a second 3x3 conv, doubling the spatial resolution.
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels,
                              kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise,
                                  use_pixel_norm, use_instance_norm, use_styles, activation_layer)

    # Upsample-conv then two epilogues to produce the next-resolution feature map.
    def forward(self, x, dlatents_in_range, noise_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0], noise_in_range[0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise_in_range[1])
        return x


class G_synthesis(nn.Module):
    # Build the synthesis network: an InputBlock plus a stack of GSynthesisBlocks
    # doubling resolution from 4x4 up to `resolution`, ending in a toRGB conv.
    def __init__(self,
                 # Disentangled latent (W) dimensionality.
                 dlatent_size=512,
                 num_channels=3,            # Number of output color channels.
                 resolution=1024,         # Output resolution.
                 # Overall multiplier for the number of feature maps.
                 fmap_base=8192,
                 # log2 feature map reduction when doubling the resolution.
                 fmap_decay=1.0,
                 # Maximum number of feature maps in any layer.
                 fmap_max=512,
                 use_styles=True,         # Enable style inputs?
                 const_input_layer=True,         # First layer is a learned constant?
                 use_noise=True,         # Enable noise inputs?
                 nonlinearity='lrelu',      # Activation function: 'relu', 'lrelu'
                 use_wscale=True,         # Enable equalized learning rate?
                 use_pixel_norm=False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm=True,         # Enable instance normalization?
                 # Low-pass filter to apply when resampling activations. None = no filtering.
                 blur_filter=[1, 2, 1],
                 ):

        super().__init__()

        # Number of feature maps at a given resolution stage (decays with depth,
        # capped at fmap_max).
        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res-1)
            name = '{s}x{s}'.format(s=2**res)
            if res == 2:
                blocks.append((name,
                               InputBlock(channels, dlatent_size, const_input_layer, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))

            else:
                blocks.append((name,
                               GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, act)))
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1,
                              gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    # Run the high-resolution tail of the network (blocks >= min_res, plus toRGB) in
    # `dtype`, leaving the cheap early blocks in fp32.
    #
    # Measured per-block on an M4 Max: cost and fp16 error are INVERSELY related. The
    # 1024/512/256 blocks are ~89% of the step time but the MOST fp16-tolerant (max image
    # error 0.006-0.011) -- their error lands near the output and doesn't propagate. The
    # small early blocks are the LEAST tolerant (8x8: 0.040; an error there is amplified
    # by every block downstream) AND are so cheap that cast overhead makes them actively
    # slower in half. So half-precision only pays at the top of the stack.
    #
    # Those blocks are contiguous, so the cast happens ONCE on entry and once on exit --
    # not per block. Casting at every boundary round-trips a 1024x1024 activation four
    # times and gives back half the speedup (measured: +4.6% vs +8.9%).
    def set_high_res_precision(self, dtype, min_res=256):
        self.half_dtype = dtype
        self.half_from_res = min_res
        for name, block in self.blocks.items():
            if int(name.split('x')[0]) >= min_res:
                block.to(dtype)
        self.torgb.to(dtype)

    # Run the latents (and per-layer noise) through every synthesis block in turn,
    # then the toRGB conv, to produce the generated image.
    def forward(self, dlatents_in, noise_in):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        half_from = getattr(self, 'half_from_res', None)
        dtype = getattr(self, 'half_dtype', None)

        for i, (name, m) in enumerate(self.blocks.items()):
            w = dlatents_in[:, 2*i:2*i+2]
            n = noise_in[2*i:2*i+2]

            # Entering the half-precision tail: cast the activation once, here. (Once, not
            # per block -- the half blocks are contiguous, and a fp16->fp32->fp16 round
            # trip at every boundary is free accuracy-wise but costs half the speedup.)
            if half_from is not None and int(name.split('x')[0]) >= half_from:
                if i > 0 and x.dtype != dtype:
                    x = x.to(dtype)
                w = w.to(dtype)
                n = [t.to(dtype) for t in n]

            if i == 0:
                x = m(w, n)
            else:
                x = m(x, w, n)

        rgb = self.torgb(x)
        return rgb.float()
