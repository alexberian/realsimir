"""Filesystem locations this project reads from.

Kept in one place so nothing downstream hard-codes a path to /workspace/data
(which idea.txt says is read-only) or to a submodule checkout.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# third-party checkouts (git submodules)
YOLO_DIR = REPO_ROOT / "pytorch-yolo-v3"
EDM2_DIR = REPO_ROOT / "edm2"

# data -- read only, never write here
DATA_ROOT = Path("/workspace/data")
ARETE_ROOT = DATA_ROOT / "arete_ir_sim_data"  # synthetic IR renders + gt metadata
OPEN_IR_ROOT = DATA_ROOT / "open_ir_images"  # real, unlabelled IR frames

__all__ = ["REPO_ROOT", "YOLO_DIR", "EDM2_DIR", "DATA_ROOT", "ARETE_ROOT", "OPEN_IR_ROOT"]
