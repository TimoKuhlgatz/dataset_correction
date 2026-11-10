"""Error 4: gaps filled without a flag in the optical reference.

What it is: a segment's orientation needs at least three measured markers of its cluster. Where fewer were seen
(occlusion), no orientation can be computed; if the deposit still carries one, the gap was filled, by the recording
software or the depositors, and nothing marks it.

How we catch it (reads only the reference): per frame, a marker counts
as measured if it has a value and was not filled with a straight line. A linear fill leaves a second difference
x[k-1] - 2 x[k] + x[k+1] of exactly zero on all three axes, up to the storage resolution of the file (rounding of
the three stored values: at most 2 resolution steps) and numerical rounding (1e-9 of the coordinate). A real marker
never moves on an exact line: on DIODEM the fills stay below 3e-12 m and measured markers above 9e-7 m. A marker held at one position
(first difference exactly zero) is a different fingerprint and is not counted as filled here. A frame whose cluster
has fewer than 3 measured markers while the reference is valid is filled without a flag. Spline or pattern fills
leave no exact line and stay invisible: a dataset without visible gaps or straight-line fills is not checkable,
not free of the error.

Straight lines are looked for only in marker files that keep full precision (GAP_FILL): in files rounded to float32
or 0.001 mm a straight fill is exact only to the rounding step, which quiet or smoothed motion reaches as well, so
there only visible gaps (missing markers) count and a fill goes unseen.

What a user of the deposit gets (the count): where the dataset publishes its own orientation or joint angles next
to the raw markers (Trento's optical quaternions, Darmstadt's joint angles, Camargo's and Central Florida walking's
OpenSim angles), a frame of a segment is unreliable if that output has a value although fewer than 3 of the
segment's markers are visible (not visible: missing, and in full-precision files also on a straight line). These
outputs are compared frame by frame with the raw markers (the provider readers below); an occurrence is a stretch of
unreliable frames. Every other dataset is checked on our reader's reference and markers with the rule above.

How we correct it: the orientation in such a frame is not measured, so the frame is marked invalid. The mask is
always built from our reader's reference and markers (GAP_FILL settings), for every trial, also where the count
comes from a provider output: that output is not what the estimators are scored against.

One occurrence is one continuous stretch of filled frames of one segment.
"""
from __future__ import annotations

import numpy as np

from ..recording import Recording
from ._result import CheckResult, entry

ERROR = 'gaps filled without a flag'
MIN_MARKERS = 3
# Numerical tolerance of "exactly on a line", relative to the coordinate. DIODEM's fills reach 1.8e-12 of the
# coordinate (2.7e-12 m; the fill passed through several float operations), real marker motion stays above 6e-7
# (9e-7 m in 0.01 % of 3.5 million marker frames): 1e-9 lies about three orders from both.
FLOAT_ROUNDING = 1e-9


def unmeasured_markers(markers: np.ndarray, resolution: float = 0.0) -> np.ndarray:
    """(N, K) bool: marker k has no value in frame i, or lies on an exactly straight line through its neighbours.

    markers: (N, K, 3) positions; resolution: the storage step of the file in the same unit (0 for full float64
    precision, 'float32' for single precision, None to count only missing values: files rounded to float32 or
    0.001 mm cannot show a straight-line fill). The first and last frame cannot be judged for
    linearity and count as measured where they have a value."""
    m = np.asarray(markers, float)
    missing = ~np.isfinite(m).all(axis=2)
    if resolution is None:
        return missing
    d1 = np.diff(m, axis=0)
    d2 = np.abs(m[:-2] - 2 * m[1:-1] + m[2:])
    with np.errstate(invalid='ignore'):
        scale = np.nanmax(np.abs(m[1:-1]), axis=2, initial=0.0) if m.shape[0] > 2 else np.zeros((0, m.shape[1]))
    relative = 2.0 ** -23 if resolution == 'float32' else 0.0  # a float32 value is exact to 2^-23 of itself
    step = 0.0 if resolution == 'float32' else float(resolution)
    tol = 2 * step + (2 * relative + FLOAT_ROUNDING) * np.nan_to_num(scale)
    on_line = np.all(d2 <= tol[..., None], axis=2) & np.isfinite(d2).all(axis=2)
    held = np.all(d1[:-1] == 0, axis=2) & np.all(d1[1:] == 0, axis=2)
    filled = np.zeros_like(missing)
    filled[1:-1] = on_line & ~held
    return missing | filled


def filled_frames(markers: np.ndarray, reference_valid: np.ndarray, resolution: float = 0.0,
                  min_markers: int = MIN_MARKERS) -> np.ndarray:
    """(N,) bool: frames where fewer than `min_markers` cluster markers were measured but the reference is valid."""
    measured = (~unmeasured_markers(markers, resolution)).sum(axis=1)
    return np.asarray(reference_valid, bool) & (measured < min_markers)


def stretches(mask: np.ndarray) -> list[tuple[int, int]]:
    """(first frame, number of frames) of every continuous run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b - a)) for a, b in zip(starts, ends)]


def _cluster(seg, need=None) -> tuple[np.ndarray | None, int]:
    """The markers the segment's frame is built from, (N, K, 3), and how many of them must be measured.

    need: (marker names or None for all, minimum measured or None for all of them). A frame fitted to the whole
    cluster needs any 3 (the default); a frame built from fixed markers (a triad, landmark formulas) needs each."""
    markers = seg.reference.markers
    if not markers:
        return None, MIN_MARKERS
    names, minimum = need or (None, MIN_MARKERS)
    names = list(markers) if names is None else list(names)
    if any(n not in markers for n in names):
        return None, MIN_MARKERS
    return np.stack([np.asarray(markers[n], float) for n in names], axis=1), (len(names) if minimum is None else minimum)


def check_for_gap_fill(recording: Recording, resolution: float = 0.0, needs: dict | None = None) -> CheckResult:
    """Find reference frames with too few measured cluster markers that are not flagged invalid.

    Each segment needs its cluster markers in `segment.reference.markers` (frame-aligned with the reference);
    a segment without markers is undecided. `resolution` is the storage step of the marker file (mm); `needs` maps
    a segment to (marker names, minimum measured), see _cluster (default: any 3 of all its markers)."""
    per_segment, decided, found = {}, False, False
    for name, seg in recording.segments.items():
        cluster, minimum = _cluster(seg, (needs or {}).get(name))
        if cluster is None or len(cluster) != len(seg.reference):
            per_segment[name] = entry(None, float('nan'), 'stretches', reason='no frame-aligned cluster markers')
            continue
        unmeasured = unmeasured_markers(cluster, resolution)
        bad = filled_frames(cluster, seg.reference.valid, resolution, minimum)
        runs = stretches(bad)
        decided = True
        found |= bool(runs)
        per_segment[name] = entry(bool(runs), len(runs), 'stretches', frames=int(bad.sum()),
                                  fraction=float(bad.mean()) if len(bad) else 0.0, markers=cluster.shape[1], needed=minimum,
                                  missing_frames=int((~np.isfinite(cluster).all(axis=2)).any(axis=1).sum()),
                                  straight_line_marker_frames=int((unmeasured & np.isfinite(cluster).all(axis=2)).sum()),
                                  longest_frames=max((n for _, n in runs), default=0))
    if not decided:
        return CheckResult(ERROR, None, 'No segment carries its cluster markers; the check cannot run.', per_segment,
                           dict(min_markers=MIN_MARKERS, resolution=resolution))
    total = sum(p['estimate'] for p in per_segment.values() if p['detected'] is not None)
    frames = sum(p['frames'] for p in per_segment.values() if p['detected'] is not None)
    summary = (f'{total} stretches ({frames} frames) carry a reference orientation although too few of the '
               'markers it is built from were measured.' if found
               else 'Every frame with a valid reference has enough measured markers.')
    return CheckResult(ERROR, found, summary, per_segment, dict(min_markers=MIN_MARKERS, resolution=resolution))


def remove_gap_fill(recording: Recording, resolution: float = 0.0, needs: dict | None = None) -> Recording:
    """Return a copy in which every reference frame with too few measured cluster markers is invalid."""
    new, marked = recording.copy(), {}
    for name, seg in new.segments.items():
        cluster, minimum = _cluster(seg, (needs or {}).get(name))
        if cluster is None or len(cluster) != len(seg.reference):
            continue
        bad = filled_frames(cluster, seg.reference.valid, resolution, minimum)
        marked[name] = int(bad.sum())
        seg.reference.valid = seg.reference.valid & ~bad
    new.log_correction(ERROR, f'Marked {sum(marked.values())} reference frames invalid that had too few measured '
                              'cluster markers.', frames_marked=marked, resolution=resolution, needs=needs)
    return new


# --- providers' own outputs against the raw markers ---------------------------------------------------------------

import io  # noqa: E402
import zipfile  # noqa: E402
from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

MIN_VISIBLE = 3
"""A provider value needs at least this many visible markers of the segment."""


def invisible_frames(markers: np.ndarray, full_precision: bool) -> np.ndarray:
    """(N,) bool: fewer than MIN_VISIBLE of the segment's markers visible. markers: (N, K, 3)."""
    unmeasured = unmeasured_markers(markers, 0.0) if full_precision else ~np.isfinite(markers).all(axis=2)
    return (~unmeasured).sum(axis=1) < MIN_VISIBLE


def provider_row(trial: str, participant: str, segment: str, invisible: np.ndarray, given: np.ndarray) -> dict:
    """One row: the stretches where the provider gives a value although too few markers are visible."""
    bad = invisible & given
    found = stretches(bad)
    return dict(trial=trial, participant=participant, segment=segment, detected=bool(found), estimate=len(found),
                unit='stretches', frames=int(bad.sum()), reference_frames=len(invisible),
                invisible=int(invisible.sum()), flagged_by_provider=int((invisible & ~given).sum()),
                longest_frames=max((n for _, n in found), default=0), source='provider output')


def _adapters():
    from ..datasets.readers.characterisation.adapters import bonfiglio, camargo, grimmer_stairs, wmcg
    return bonfiglio, camargo, grimmer_stairs, wmcg


TRENTO_QUATERNIONS = {'trunk': ['TH'], 'scapula': ['SC'], 'upper_arm': ['UA1', 'UA2'], 'forearm': ['LA'],
                      'hand': ['HA']}
"""Trento: the provider's optical quaternions (qGS_Optotrack) per segment."""


def trento_subject(loader, subject: str) -> list[dict]:
    """Trento: every raw optical file of the subject against the provider's quaternions, matched by order and
    length (a file without a provider trial of the same length is not compared). Trials are the optical files."""
    import scipy.io as sio
    trento = _adapters()[0]
    a, rows = trento.BonfiglioAdapter(), []
    q = np.atleast_1d(sio.loadmat(str(a._subject_dir(subject) / '30. Results' / 'qGS_Optotrack'),
                                  squeeze_me=True)['qGS'])
    qi = 0
    for k, f in enumerate(a._optical_files(subject)):
        mk = trento._read_optical_markers(f)
        n = len(next(v for v in mk.values() if v is not None))
        j = next((j for j in range(qi, len(q)) if len(np.atleast_2d(q[j]['TH'])) == n), None)
        trial = f'{subject}/optical-{k + 1:02d}'
        if j is None:
            rows.append(dict(trial=trial, participant=subject, segment='', detected=None,
                             reason='no provider trial of this length'))
            continue
        qi = j + 1
        for seg, (_, names) in trento.SEGMENTS.items():
            if any(mk.get(x) is None for x in names):
                continue
            given = np.zeros(n, bool)
            for qn in TRENTO_QUATERNIONS[seg]:
                qq = np.atleast_2d(q[j][qn]).astype(float)
                given |= np.isfinite(qq).all(1) & (np.abs(qq).sum(1) > 0)
            rows.append(provider_row(trial, subject, seg,
                                     invisible_frames(np.stack([mk[x] for x in names], axis=1), True), given))
    return rows


DARMSTADT_ANGLES = {'thigh': ['hip_angle_flexion', 'hip_angle_adduction', 'hip_angle_rotation', 'knee_angle_flexion'],
                    'shank': ['knee_angle_flexion', 'ankle_angle_flexion'], 'foot': ['ankle_angle_flexion']}
"""Darmstadt: the provider's joint angles (Processed/Joint_Kinematics) that involve each segment."""


def darmstadt_subject(loader, subject: str) -> list[dict]:
    """Darmstadt: the joint angles against the markers our reader carries (full precision, straight lines tested)."""
    import scipy.io as sio
    adapter = _adapters()[2].GrimmerStairsAdapter()
    sn = int(subject.lstrip('S'))
    with zipfile.ZipFile(adapter.root / 'Processed.zip') as z:
        kinematics = sio.loadmat(io.BytesIO(z.read(f'Processed/Joint_Kinematics/Joint_Kinematics{sn}.mat')),
                                 squeeze_me=True, struct_as_record=False)['Joint_Kinematics']
    rows = []
    for trial in [t for t in loader.trials() if str(loader.participant_of(t)) == subject]:
        _, cfg, rep = adapter._parse_trial_id(*trial.split('/'))
        rec, angles = loader.load(trial), kinematics[cfg][rep]
        for seg, sg in rec.segments.items():
            part, side = seg.rsplit('_', 1)
            m = np.stack(list(sg.reference.markers.values()), axis=1)
            values = [np.asarray(getattr(angles, f'{x}_{side}'), float) for x in DARMSTADT_ANGLES[part]]
            if any(len(v) != len(m) for v in values):
                rows.append(dict(trial=trial, participant=subject, segment=seg, detected=None,
                                 reason='provider angles and markers differ in length'))
                continue
            given = np.zeros(len(m), bool)
            for v in values:
                given |= np.isfinite(v)
            rows.append(provider_row(trial, subject, seg, invisible_frames(m, True), given))
    return rows


CAMARGO_ANGLES = {'thigh': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r'],
                  'shank': ['knee_angle_r', 'ankle_angle_r'], 'foot': ['ankle_angle_r', 'subtalar_angle_r']}
"""Camargo: the provider's OpenSim angles (ik table) that involve each segment."""


def camargo_trial(loader, trial: str) -> list[dict]:
    """Camargo: the ik table against the raw markers, on the same clock (float32 files: visible gaps only)."""
    camargo = _adapters()[1]
    a = camargo.Camargo()
    subject, name = trial.split('/', 1)
    mk, ik = a._table(subject, name, 'markers'), a._table(subject, name, 'ik')
    if mk is None or ik is None:
        return [dict(trial=trial, participant=subject, segment='', detected=None,
                     reason='no markers' if mk is None else 'no ik table')]
    t_m, t_ik = np.asarray(mk['Header'], float), np.asarray(ik['Header'], float)
    idx = np.searchsorted(t_ik, t_m)
    ok = (idx < len(t_ik)) & (np.abs(t_ik[np.clip(idx, 0, len(t_ik) - 1)] - t_m) < 1e-6)
    rows = []
    for seg, names in camargo.MARKERS.items():
        m = np.stack([np.stack([mk[f'{x}_{c}'] for c in 'xyz'], axis=1) for x in names], axis=1).astype(float)
        given = np.zeros(len(t_m), bool)
        for col in CAMARGO_ANGLES[seg]:
            v = np.asarray(ik[col], float)
            given |= ok & np.isfinite(v[np.clip(idx, 0, len(v) - 1)])
        rows.append(provider_row(trial, subject, seg, invisible_frames(m, False), given))
    return rows


def wmcg_angles(segment: str) -> list[str]:
    """Central Florida walking: the OpenSim (gait2354) coordinates that involve each segment."""
    side = segment.split('_')[0][0] if '_' in segment else None
    if segment == 'sternum':
        return ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    if segment == 'sacrum':
        return ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'lumbar_extension', 'lumbar_bending',
                'lumbar_rotation'] + [f'hip_{x}_{s}' for x in ('flexion', 'adduction', 'rotation') for s in 'rl']
    part = segment.split('_')[1]
    return {'thigh': [f'hip_flexion_{side}', f'hip_adduction_{side}', f'hip_rotation_{side}', f'knee_angle_{side}'],
            'shank': [f'knee_angle_{side}', f'ankle_angle_{side}'],
            'foot': [f'ankle_angle_{side}', f'subtalar_angle_{side}', f'mtp_angle_{side}']}[part]


def wmcg_all(loader, _=None) -> list[dict]:
    """Central Florida walking: the /kinematics table against the raw markers of every recording (float32 files:
    visible gaps only); the kinematics run on a 100 Hz clock of their own."""
    import h5py
    wmcg = _adapters()[3]
    decode = lambda xs: [x.decode() if isinstance(x, bytes) else str(x) for x in xs]  # noqa: E731
    rows = []
    with h5py.File(wmcg.Wmcg().path, 'r') as h5:
        for p in h5['participants']:
            for mode in h5['participants'][p]:
                for cond in h5['participants'][p][mode]:
                    g = h5['participants'][p][mode][cond]
                    if 'markers' not in g or 'kinematics' not in g:
                        continue
                    names = decode(g['markers/marker_names'][:])
                    data = np.asarray(g['markers/data'], float).reshape(len(g['markers/data']), -1, 3)
                    t_m, t_k = np.asarray(g['markers/time'], float), np.asarray(g['kinematics/time'], float)
                    joints = decode(g['kinematics/joint_names'][:])
                    kin = np.asarray(g['kinematics/data'], float)
                    idx = np.clip(np.round(t_m * 100).astype(int), 0, len(t_k) - 1)
                    ok = np.abs(t_k[idx] - t_m) < 1e-4
                    for seg, mnames in wmcg.MARKERS.items():
                        if any(x not in names for x in mnames):
                            continue
                        given = np.zeros(len(t_m), bool)
                        for x in wmcg_angles(seg):
                            given |= ok & np.isfinite(kin[idx, joints.index(x)])
                        rows.append(provider_row(f'{p}/{mode}/{cond}', str(p), seg,
                                                 invisible_frames(data[:, [names.index(x) for x in mnames]], False),
                                                 given))
    return rows


def _subjects(loader) -> list[str]:
    return list(dict.fromkeys(str(loader.participant_of(t)) for t in loader.trials()))


PROVIDERS = {
    'bonfiglio': (trento_subject, lambda loader: sorted({s for s, _ in _adapters()[0].BonfiglioAdapter().trials()},
                                                         key=int)),
    'grimmer_stairs': (darmstadt_subject, _subjects),
    'camargo': (camargo_trial, lambda loader: list(loader.trials())),
    'wmcg': (wmcg_all, lambda loader: [None]),
}
"""Datasets counted on their provider's own output: (function(loader, item) -> rows, items to run it on)."""


# --- handler: one dataset at a time -------------------------------------------------------------------------------

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import parallel_map, reader_of, truthy  # noqa: E402

ALL = 'all'
GAP_FILL = {'diodem': (0.0, None), 'grimmer_stairs': (0.0, ALL), 'bonfiglio': (0.0, ALL),
            'bonfiglio_deposited': (0.0, ALL)}
"""Per reader, for the mask and the datasets checked on our reader: (storage resolution, markers needed) where the
marker files keep full precision and straight-line fills can be seen. ALL = every marker of the frame is needed (a
frame built from fixed landmarks or a three-marker cluster); None = any three of the segment's markers. Every other
dataset with markers: only missing markers count (resolution None), any three of the segment's markers."""


def settings(reader: str, rec: Recording) -> tuple:
    """(resolution, needs) of check_for_gap_fill and remove_gap_fill for a recording of `reader`."""
    resolution, rule = GAP_FILL.get(reader, (None, None))
    return resolution, ({s: (None, None) for s in rec.segments} if rule == ALL else None)


def _check_trial(loader, trial, reader: str) -> list[dict]:
    rec = loader.load(trial)
    resolution, needs = settings(reader, rec)
    rows = []
    for seg, p in check_for_gap_fill(rec, resolution=resolution, needs=needs).per_segment.items():
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), segment=seg,
                         detected=p['detected'], estimate=p['estimate'], unit='stretches', frames=p.get('frames'),
                         reference_frames=len(rec.segments[seg].reference), markers=p.get('markers'),
                         needed=p.get('needed'), missing_frames=p.get('missing_frames'),
                         straight_line_marker_frames=p.get('straight_line_marker_frames'),
                         longest_frames=p.get('longest_frames'), reason=p.get('reason', ''), source='reader'))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the stretches of frames where the dataset gives an orientation although too
    few markers were measured (estimate). Trento, Darmstadt, Camargo and Central Florida walking: their provider's
    own output against the raw markers (PROVIDERS; Trento's trials are its optical files). Every other dataset: our
    reader's reference and markers; undecided where a segment carries no frame-aligned cluster markers."""
    reader = reader_of(loader)
    name = 'bonfiglio' if dataset == 'bonfiglio_deposited' else dataset
    if name in PROVIDERS:
        fn, items = PROVIDERS[name]
        jobs = items(loader)
    else:
        fn, jobs = partial(_check_trial, reader=reader), loader.trials()
    rows = []
    for out in parallel_map(loader, fn, jobs, processes):
        rows += [dict(out, segment='', detected=None)] if isinstance(out, dict) else out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with an unflagged filled stretch / such stretches."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'stretches', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _mask(rec: Recording, trial, *, reader: str) -> Recording:
    if not any(seg.reference.markers for seg in rec.segments.values()):
        return rec
    resolution, needs = settings(reader, rec)
    return remove_gap_fill(rec, resolution=resolution, needs=needs)


def correct(loader, rows: pd.DataFrame):
    """The frames with too few measured markers marked invalid, from our reader's reference and markers (GAP_FILL)
    in every trial (module description); recomputed on loading, since the mask depends only on the trial."""
    return CorrectedLoader(loader, partial(_mask, reader=reader_of(loader)), 'gaps filled without a flag')


SPEC = ErrorSpec(4, 'gap_fill', 'mask', 'gaps filled without a flag in the optical reference')
PARTS = {4: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_gap_fill = make_handler(PARTS[4])
