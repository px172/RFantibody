from contextlib import contextmanager

import torch

if torch.cuda.is_available():
    from torch.cuda.nvtx import range as nvtx_range
else:

    @contextmanager
    def nvtx_range(_name):
        yield
