# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Backend-agnostic NVTX range shim. NVTX ranges are pure profiling annotations
# with no effect on results, but ``torch.cuda.nvtx`` is only meaningful on a
# CUDA/ROCm device and can be unavailable on Intel XPU / CPU. This uses real
# ranges when a CUDA/ROCm device is active (so Nsight profiling still works) and
# is a no-op otherwise.

import contextlib

import torch


@contextlib.contextmanager
def nvtx_range(msg: str):
    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(msg):
            yield
    else:
        yield
