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


def sync_device():
    # Block until the device has finished the work already queued on it.
    #
    # CUDA and MPS both dispatch asynchronously: Python enqueues kernels and returns
    # immediately. Any wall-clock timing that doesn't sync first measures the ENQUEUE
    # rate, not the compute -- which silently reports impossible throughput. Call this
    # before stopping a timer. No-op on CPU, which is already synchronous.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


device = get_device()
