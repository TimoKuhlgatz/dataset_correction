"""The OpenSense reference reconstruction. The deposit's marker-plate orientation file is defective (its left and
right foot columns are permutations of one another), so the reference is rebuilt from the markers and the raw
gyroscopes. :func:`~mocapref.datasets.readers.datasets.opensense.reference.build_trial` is the whole pipeline;
:mod:`~mocapref.datasets.readers.datasets.opensense.store` keeps its output in a data home so it is built once.
"""
from .reference import build_trial, diagnostics_of, marker_file_path, plate_orientations
from .sensors import read_sensors
from .store import TRIALS, load_trial, rebuilt_path

__all__ = ["TRIALS", "build_trial", "diagnostics_of", "load_trial", "marker_file_path", "plate_orientations",
           "read_sensors", "rebuilt_path"]
