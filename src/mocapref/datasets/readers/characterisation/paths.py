"""Where the data live. Every path can be overridden through the environment variable named in ``ENV``."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()
RAW = Path("/home/data/imu-orientation/raw")
PREPARED = Path("/home/data/imu-orientation/prepared")
# The data homes the readers use; the environment variables override them.
ENV = {
    "eval_datasets": ("MOCAPREF_DATA", RAW),
    "opensense": ("MOCAPREF_OPENSENSE", RAW / "opensense/_extracted"),
    # The OpenSense reference rebuilt from the markers (generated data), served without the constant-lag removal
    # (datasets/opensense/reference.py, REMOVE_CONSTANT_LAG).
    "opensense_reference": ("MOCAPREF_OPENSENSE_REFERENCE", PREPARED / "opensense/reference_as_deposited"),
}


def data_path(key: str) -> Path:
    """Resolve a data home by key; the environment variable wins over the default."""
    env, default = ENV[key]
    return Path(os.environ.get(env, default)).expanduser()


def dataset_dir(name: str) -> Path:
    """The folder of one downloaded dataset under the raw-data home."""
    return data_path("eval_datasets") / name
