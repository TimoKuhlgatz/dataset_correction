"""Where the rebuilt OpenSense reference lives, and how it is (re)built.

The reconstruction (:mod:`.reference`) reads about a gigabyte of text per trial and takes minutes, so each trial is
built once into its own data home (``paths.data_path("opensense_reference")``, outside the repository) and read
from there afterwards; deleting a file rebuilds it from the deposit.

One file per trial: ``quat__<sensor>``, ``acc__<sensor>``, ``gyr__<sensor>`` (float64), ``imus`` (the sensor
names), ``n`` (the sample count) and ``native_rate`` (the trial's sensor rate). The trial's diagnostics are written
beside it as ``<Subject>_<task>_diag.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ...characterisation.paths import data_path
from .deposit_files import trial_folder
from .reference import build_trial, diagnostics_of, marker_file_path

TRIALS: tuple[tuple[str, str], ...] = (
    ("Subject01", "walking"), ("Subject01", "complexTasks"),
    ("Subject02", "walking"), ("Subject02", "complexTasks"),
    ("Subject03", "walking"), ("Subject03", "complexTasks"),
    ("Subject04", "walking"), ("Subject04", "complexTasks"),
    ("Subject05", "walking"),
    ("Subject06", "walking"), ("Subject06", "complexTasks"),
    ("Subject07", "walking"), ("Subject07", "complexTasks"),
    ("Subject08", "walking"), ("Subject08", "complexTasks"),
    ("Subject09", "walking"), ("Subject09", "complexTasks"),
    ("Subject10", "walking"),
    ("Subject11", "walking"), ("Subject11", "complexTasks"),
)
"""Every subject-and-task folder of the deposit that ships a marker file — twenty of them. Not all reconstruct:
three ship fewer than two usable sensors. Which ones do is a measurement, written into the store's own
``index.json`` by :func:`build_all`, never assumed here."""


def deposit_root() -> Path:
    """The deposit's home (``configs/data_paths.yaml``: ``opensense``)."""
    return data_path("opensense")


def store_dir() -> Path:
    """The home of the rebuilt trials (``configs/data_paths.yaml``: ``opensense_reference``)."""
    return data_path("opensense_reference")


def rebuilt_path(subject: str, task: str, store: Path | None = None) -> Path:
    return (store_dir() if store is None else Path(store)) / f"{subject}_{trial_folder(task)}.npz"


def diagnostics_path(subject: str, task: str, store: Path | None = None) -> Path:
    return (store_dir() if store is None else Path(store)) / f"{subject}_{trial_folder(task)}_diag.json"


def write_trial(trial: dict, store: Path | None = None) -> Path:
    """Write one reconstructed trial and its diagnostics into the store; returns the array file's path."""
    subject, task = trial["subject"], trial["task"]
    target = rebuilt_path(subject, task, store)
    target.parent.mkdir(parents=True, exist_ok=True)
    # written under a process-unique name and moved into place, so that several scorer workers building
    # different trials at once can never observe a half-written file
    staging = target.with_name(f"{target.stem}.partial-{os.getpid()}.npz")
    np.savez_compressed(
        staging,
        **{f"quat__{k}": v.astype(np.float64) for k, v in trial["quat"].items()},
        **{f"acc__{k}": v.astype(np.float64) for k, v in trial["acc"].items()},
        **{f"gyr__{k}": v.astype(np.float64) for k, v in trial["gyr"].items()},
        imus=np.array(trial["imus"]),
        n=np.array(trial["n"]),
        native_rate=np.array(trial["recovery"]["native_rate"]),
    )
    staging.replace(target)
    diagnostics_path(subject, task, store).write_text(
        json.dumps(diagnostics_of(trial), indent=1, default=float))
    return target


def build(subject: str, task: str, root: Path | None = None, store: Path | None = None,
          verbose: bool = False) -> dict:
    """Rebuild one trial from the deposit and write it into the store; returns the diagnostics.

    A trial the reconstruction refuses (no marker file, fewer than two usable sensors) is returned with its
    ``status`` and nothing is written.
    """
    trial = build_trial(subject, task, deposit_root() if root is None else Path(root), verbose=verbose)
    if trial["status"] != "ok":
        return trial
    write_trial(trial, store)
    return diagnostics_of(trial)


def load_trial(subject: str, task: str, store: Path | None = None, build_if_missing: bool = True) -> dict:
    """One rebuilt trial as arrays: ``{'imus', 'n', 'native_rate', 'quat', 'acc', 'gyr'}``.

    Builds it from the deposit first if the store does not hold it yet.
    """
    path = rebuilt_path(subject, task, store)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"{path} is not in the rebuilt-reference store")
        outcome = build(subject, task, store=store)
        if outcome.get("status") != "ok":
            raise FileNotFoundError(
                f"{subject}/{task} cannot be reconstructed from {deposit_root()}: {outcome.get('status')}")
    with np.load(path) as store_file:
        names = [str(x) for x in store_file["imus"]]
        return {"imus": names,
                "n": int(store_file["n"]),
                "native_rate": float(store_file["native_rate"]),
                "quat": {k: store_file[f"quat__{k}"] for k in names},
                "acc": {k: store_file[f"acc__{k}"] for k in names},
                "gyr": {k: store_file[f"gyr__{k}"] for k in names}}


def build_all(root: Path | None = None, store: Path | None = None, verbose: bool = True) -> dict:
    """Rebuild every trial of :data:`TRIALS` and write an ``index.json`` naming which reconstructed."""
    deposit = deposit_root() if root is None else Path(root)
    index = {}
    for subject, task in TRIALS:
        if marker_file_path(deposit / subject, task) is None:
            index[f"{subject}/{task}"] = {"status": "no trc"}
            continue
        outcome = build(subject, task, root=deposit, store=store, verbose=False)
        index[f"{subject}/{task}"] = {
            "status": outcome.get("status"),
            "n": outcome.get("n"),
            "imus": outcome.get("imus"),
            "native_rate": outcome.get("recovery", {}).get("native_rate"),
            "trial_shift_samples": outcome.get("recovery", {}).get("trial_shift_samples"),
        }
        if verbose:
            print(f"{subject}/{task}: {index[f'{subject}/{task}']}")
    target = (store_dir() if store is None else Path(store)) / "index.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"deposit": str(deposit), "trials": index}, indent=1, default=float))
    return index
