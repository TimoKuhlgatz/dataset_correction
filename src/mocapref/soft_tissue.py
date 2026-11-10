"""Soft-tissue motion: how far an IMU and its skin-marker segment frame move against each other.

Soft-tissue motion is not one of the numbered errors: it is a property of every skin-marker reference. It can be
measured where a dataset carries two optical frames per segment: a marker cluster rigidly fixed on the IMU housing
(the IMU's own orientation) and a segment frame from skin markers or a skeletal model. Three datasets do:
TUK six-minute walk (tuk6min), Geneva gait (grouvel) and Hankyong full body (fullbody17).

Segment level. Per trial and segment, q_err(t) = q_segment(t)^-1 * q_housing(t), the IMU's orientation in the
segment frame. Its constant part (the trial's chordal mean rotation) is mostly the frame definitions and is not
interpreted; the time-varying part, the angle of q_err(t) against that mean, is the soft-tissue motion.

Joint level ("floor"). For each joint, the relative orientation of the two IMU housings, with each segment's constant
part removed (a perfect calibration), is compared with the same joint from the two skin-marker frames; the mean angle
over the trial is the error that a perfect IMU estimate would still show against the skin-marker reference.

    from mocapref.soft_tissue import segment_motion, joint_floor, summarize
    seg = segment_motion('tuk6min')        # one row per trial and segment
    jnt = joint_floor('tuk6min')           # one row per trial and joint
    print(summarize(seg, jnt))

Main findings (measured on every trial of the three datasets, exploratory): the time-varying part is typically
1-5 deg per segment (median per trial) and 3-17 deg at the 95th percentile, largest for the Geneva thigh (median
4.6 deg, 95th percentile 17.1 deg) and the Hankyong upper arm (6.2 deg, 17.4 deg). The joint-level floor, as the mean
over a trial's joints, is 4.3 deg (TUK), 6.2 deg (Geneva) and 4.9 deg (Hankyong), and 6-8 deg for hip, knee and elbow
in some datasets. An IMU orientation estimate cannot be expected to agree with a skin-marker reference better than
this. Single extreme trials (Geneva P10/Sitting_01 foot, P01/Sitting_01 lumbar; Hankyong S08/Squat_Jump_1 elbow)
have not been checked for marker defects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DATASETS = {'tuk6min': 'TUK six-minute walk', 'grouvel': 'Geneva gait', 'fullbody17': 'Hankyong full body'}
_READER = {'grouvel': 'grouvel-geneva', 'fullbody17': 'fullbody-17imu'}
MIN_FRAMES = 20


def _trials(dataset: str) -> list:
    if dataset == 'tuk6min':
        from .datasets import load_dataset
        return load_dataset('tuk6min').trials()
    from .datasets.readers.characterisation.adapters import adapter
    return list(adapter(_READER[dataset]).trials())


def frames(dataset: str, trial) -> tuple[dict, list]:
    """({segment: (housing (N,3,3), skin (N,3,3))}, joints as (parent, child)) of one trial, on the housing's frames."""
    if dataset == 'tuk6min':
        from .datasets import load_dataset
        rec = load_dataset('tuk6min').load(trial)
        out = {}
        for name, seg in rec.segments.items():
            h = seg.reference.orientation.as_matrix()
            s = rec.metadata['anatomical_frames'][name].as_matrix()
            n = min(len(h), len(s))
            out[name] = (h[:n], s[:n])
        return out, list(rec.joints)
    from .datasets.readers.characterisation.adapters import adapter
    rec = adapter(_READER[dataset]).load(*trial)
    out = {}
    for name, seg in rec.segments.items():
        h, a = seg.reference, seg.anatomical
        if h is None or a is None or h.kind != 'housing':
            continue
        Ra = a.R
        if len(h.t) != len(a.t) or not np.allclose(h.t, a.t):  # one optical clock: nearest frame
            idx = np.clip(np.round(np.interp(h.t, a.t, np.arange(len(a.t)))).astype(int), 0, len(a.t) - 1)
            Ra = a.R[idx]
        out[name] = (h.R, Ra)
    return out, [(p.parent, p.child) for p in rec.pairs]


def _finite(*mats) -> np.ndarray:
    return np.logical_and.reduce([np.isfinite(m).all((1, 2)) for m in mats])


def segment_error(housing: np.ndarray, skin: np.ndarray) -> dict | None:
    """Constant and time-varying part of q_skin^-1 * q_housing for one segment (None with too few frames)."""
    ok = _finite(housing, skin)
    if ok.sum() < MIN_FRAMES:
        return None
    err = Rotation.from_matrix(skin[ok]).inv() * Rotation.from_matrix(housing[ok])
    mean = err.mean()
    dev = np.degrees((mean.inv() * err).magnitude())
    return dict(frames=int(ok.sum()), constant_angle_deg=float(np.degrees(mean.magnitude())),
                dev_median_deg=float(np.median(dev)), dev_p95_deg=float(np.percentile(dev, 95)),
                dev_rms_deg=float(np.sqrt(np.mean(dev ** 2))), dev_max_deg=float(dev.max()))


def joint_error(ha: np.ndarray, sa: np.ndarray, hb: np.ndarray, sb: np.ndarray) -> np.ndarray | None:
    """Per frame, the angle (deg) between the housing joint with each segment's constant part removed and the
    skin-marker joint (None with too few frames)."""
    n = min(len(ha), len(hb), len(sa), len(sb))
    ha, sa, hb, sb = ha[:n], sa[:n], hb[:n], sb[:n]
    ok = _finite(ha, sa, hb, sb)
    if ok.sum() < MIN_FRAMES:
        return None
    ha, sa, hb, sb = (Rotation.from_matrix(x[ok]) for x in (ha, sa, hb, sb))
    ca, cb = (sa.inv() * ha).mean(), (sb.inv() * hb).mean()
    imu_joint = (ha * ca.inv()).inv() * (hb * cb.inv())
    return np.degrees(((sa.inv() * sb).inv() * imu_joint).magnitude())


def _trial_name(trial) -> str:
    return trial if isinstance(trial, str) else '/'.join(trial)


def _segment_rows(dataset, trial):
    try:
        fr, _ = frames(dataset, trial)
    except Exception as exc:  # noqa: BLE001
        return [dict(dataset=dataset, trial=_trial_name(trial), error=repr(exc)[:150])]
    rows = []
    for name, (h, s) in fr.items():
        r = segment_error(h, s)
        if r is not None:
            rows.append(dict(dataset=dataset, trial=_trial_name(trial), segment=name, **r))
    return rows


def _joint_rows(dataset, trial):
    try:
        fr, joints = frames(dataset, trial)
    except Exception as exc:  # noqa: BLE001
        return [dict(dataset=dataset, trial=_trial_name(trial), error=repr(exc)[:150])]
    rows = []
    for parent, child in joints:
        if parent not in fr or child not in fr:
            continue
        e = joint_error(fr[parent][0], fr[parent][1], fr[child][0], fr[child][1])
        if e is not None:
            rows.append(dict(dataset=dataset, trial=_trial_name(trial), parent=parent, child=child, frames=len(e),
                             mean_deg=float(e.mean()), median_deg=float(np.median(e)),
                             p95_deg=float(np.percentile(e, 95)), max_deg=float(e.max())))
    return rows


def _run(fn, datasets, processes):
    datasets = list(DATASETS) if datasets is None else ([datasets] if isinstance(datasets, str) else datasets)
    jobs = [(d, t) for d in datasets for t in _trials(d)]
    if processes > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(processes) as pool:
            parts = list(pool.map(fn, *zip(*jobs)))
    else:
        parts = [fn(d, t) for d, t in jobs]
    return pd.DataFrame([r for p in parts for r in p])


def segment_motion(datasets=None, processes: int = 1) -> pd.DataFrame:
    """One row per trial and segment: constant angle and time-varying part (median, 95th percentile, RMS, max)."""
    return _run(_segment_rows, datasets, processes)


def joint_floor(datasets=None, processes: int = 1) -> pd.DataFrame:
    """One row per trial and joint: mean, median, 95th percentile and max of the joint-level floor."""
    return _run(_joint_rows, datasets, processes)


def _side_free(name: str) -> str:
    for p in ('left_', 'right_', 'l_', 'r_'):
        if name.startswith(p):
            return name[len(p):]
    return name


def summarize(segments: pd.DataFrame, joints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per dataset and segment (left and right pooled): median over trials of each trial's median and 95th
    percentile; per joint: median over trials of each trial's mean, plus the mean over a trial's joints."""
    seg_rows, joint_rows = [], []
    segments = segments.dropna(subset=['segment']) if 'segment' in segments else segments.iloc[:0]
    joints = joints.dropna(subset=['parent']) if 'parent' in joints else joints.iloc[:0]
    for ds, s in segments.groupby('dataset', sort=False):
        for body, g in s.groupby(s.segment.map(_side_free), sort=False):
            seg_rows.append(dict(dataset=DATASETS.get(ds, ds), segment=body, trials=g.trial.nunique(),
                                 time_varying_median_deg=g.dev_median_deg.median(),
                                 time_varying_p95_deg=g.dev_p95_deg.median(), worst_trial_p95_deg=g.dev_p95_deg.max()))
    for ds, j in joints.groupby('dataset', sort=False):
        name = j.parent.map(_side_free) + '-' + j.child.map(_side_free)
        for joint, g in j.groupby(name, sort=False):
            joint_rows.append(dict(dataset=DATASETS.get(ds, ds), joint=joint, trials=g.trial.nunique(),
                                   mean_deg=g.mean_deg.median(), p95_deg=g.p95_deg.median()))
        per_trial = j.groupby('trial').mean_deg.mean()
        joint_rows.append(dict(dataset=DATASETS.get(ds, ds), joint='all joints of a trial', trials=per_trial.size,
                               mean_deg=per_trial.median(), p95_deg=np.nan))
    return pd.DataFrame(seg_rows), pd.DataFrame(joint_rows)
