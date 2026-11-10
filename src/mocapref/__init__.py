"""Defects and reference accuracy of public IMU datasets with an optical motion-capture reference.

Package layout:
    recording.py         the common data structure: Recording -> Segment -> ImuStream + ReferenceStream
    datasets/            one loader per dataset, all returning Recording objects
    errors/              one module per error, each with a check and a removal
    correction.py        correct_in_order(): every error corrected in the agreed order
    effect.py            the effect of an error on relative orientation estimation (MEKF or zoe)
    effect_analysis.py   the effect analysis over the datasets with a saved tuning split
    tuning_split.py      the tuning/test split of each dataset (saved in splits/)
    motion_onset.py      where a trial starts to move, and from when it is scored
    soft_tissue.py       soft-tissue motion between marker cluster and IMU
    zoe.py               the calibration-free relative orientation estimator
    resample.py          common time grid for IMU and reference
    rotations.py         shared rotation and signal helpers
    synthetic.py         synthetic recordings with known injected errors, for the known-answer tests
"""
from .recording import Correction, ImuStream, Recording, ReferenceStream, Segment

__all__ = ['Recording', 'Segment', 'ImuStream', 'ReferenceStream', 'Correction']
