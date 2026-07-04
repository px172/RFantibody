"""Backend-agnostic device helpers.

Lets RFantibody run on NVIDIA CUDA, AMD ROCm (which PyTorch exposes through the
same ``torch.cuda`` API via HIP), Intel XPU, or CPU without hardcoding
``"cuda"``. Prefer :func:`get_device` for device selection and the autocast /
memory helpers below over calling ``torch.cuda.*`` directly, so the same code
runs on any backend.
"""
import functools

import torch


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_device(prefer=None) -> torch.device:
    """Return the best available device.

    ``prefer`` (e.g. a value from a CLI flag or config) is honored verbatim
    when given. Otherwise pick CUDA/ROCm, then Intel XPU, then CPU.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():          # NVIDIA CUDA or AMD ROCm (HIP)
        return torch.device("cuda")
    if _xpu_available():                    # Intel XPU
        return torch.device("xpu")
    return torch.device("cpu")


def _accelerator_module(device_type: str):
    """Return the ``torch.cuda`` / ``torch.xpu`` submodule for a device type, or None."""
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda
    if device_type == "xpu" and _xpu_available():
        return torch.xpu
    return None


def autocast(device, enabled: bool = True, **kwargs):
    """Device-agnostic replacement for ``torch.cuda.amp.autocast``."""
    return torch.autocast(device_type=torch.device(device).type, enabled=enabled, **kwargs)


def _infer_device(module, args) -> torch.device:
    for p in getattr(module, "parameters", lambda: [])():
        return p.device
    for a in args:
        if torch.is_tensor(a):
            return a.device
    return torch.device("cpu")


def autocast_disabled(func):
    """Decorator forcing full precision on whichever backend the call runs on.

    Portable replacement for ``@torch.cuda.amp.autocast(enabled=False)`` applied
    to an ``nn.Module.forward``. The device is inferred from the module's
    parameters, falling back to the first tensor argument, then CPU.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        device = _infer_device(self, args)
        with torch.autocast(device_type=device.type, enabled=False):
            return func(self, *args, **kwargs)

    return wrapper


def empty_cache(device) -> None:
    """Release cached memory on ``device``'s accelerator (no-op on CPU)."""
    m = _accelerator_module(torch.device(device).type)
    if m is not None:
        m.empty_cache()


def reset_peak_memory_stats(device) -> None:
    m = _accelerator_module(torch.device(device).type)
    if m is not None:
        m.reset_peak_memory_stats()


def max_memory_allocated(device) -> int:
    """Peak bytes allocated on ``device``'s accelerator (0 on CPU)."""
    m = _accelerator_module(torch.device(device).type)
    return m.max_memory_allocated() if m is not None else 0
