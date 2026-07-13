import os
import torch


def get_device():
    # Pick the best available torch device.
    #
    # Order: CUDA, then Apple MPS (Metal), then CPU. Override with the
    # PULSE_DEVICE env var (e.g. PULSE_DEVICE=cpu) if MPS hits an
    # unsupported op and you want to force a fallback.
    forced = os.environ.get("PULSE_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
