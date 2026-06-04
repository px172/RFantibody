import os

import torch


def _backend_available(backend: str) -> bool:
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if backend == "cpu":
        return True
    return False


def get_accelerator_device() -> torch.device:
    requested = os.getenv("RFANTIBODY_DEVICE")
    if requested:
        backend = requested.split(":", 1)[0].lower()
        if not _backend_available(backend):
            raise RuntimeError(f"Requested RFANTIBODY_DEVICE={requested}, but {backend} is not available")
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def seed_accelerators(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def describe_device(device: torch.device) -> str:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device.index or torch.cuda.current_device())
    if device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        get_device_name = getattr(torch.xpu, "get_device_name", None)
        if get_device_name is not None:
            return get_device_name(device.index or torch.xpu.current_device())
        return "Intel XPU"
    return str(device).upper()
