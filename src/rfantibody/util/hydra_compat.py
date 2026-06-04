"""Compatibility shims for Hydra versions used by RFantibody."""

from __future__ import annotations

import argparse
import sys
from typing import Any


_PATCHED_ARGPARSE_HELP = False


def patch_hydra_argparse_help() -> None:
    """Allow Hydra 1.3 lazy help objects under Python 3.14 argparse.

    Python 3.14 validates argparse help strings when arguments are added.
    Hydra 1.3 passes a lazy completion help object instead of a plain string,
    which raises before the RFantibody config is parsed.
    """

    global _PATCHED_ARGPARSE_HELP
    if _PATCHED_ARGPARSE_HELP or sys.version_info < (3, 14):
        return

    original_check_help = argparse.ArgumentParser._check_help

    def check_help(self: argparse.ArgumentParser, action: Any) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        original_check_help(self, action)

    argparse.ArgumentParser._check_help = check_help
    _PATCHED_ARGPARSE_HELP = True
