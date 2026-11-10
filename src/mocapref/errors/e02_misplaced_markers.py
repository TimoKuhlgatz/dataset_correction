"""Error 2: a marker of a cluster reconstructed in the wrong place while the cluster is visible.

What it is: every marker of the cluster has a position and its label is right, but one marker's position is wrong
for a few frames (a reflection or ghost taken for it, two markers merged, poor triangulation). The cluster is
dented and the orientation built from it tilts or jumps. Only transient errors are looked for: the marker returns to
its place within the trial. A run that reaches the trial's first or last frame (a possible marker that came loose or
was re-glued) is reported, not counted.

How we catch it (reads only the reference, after the label swaps of error 1). The cluster's shape (marker positions
relative to each other) is learnt from the trial's own frames (iterated least-squares alignment, median positions).
In every frame with at least four markers, each marker k is left out once and the shape is fitted to the others. Two
numbers per marker: its displacement, the distance from where the other markers place it (mm), and its influence,
the angle between the orientations fitted with and without it (degrees). The marker to blame is the one whose
leave-out fit fits the others best. It is misplaced if both its displacement and its influence lie above the normal
range of the trial: Tukey's far-out fence (upper quartile + 3 IQR) over all markers and frames of the trial; the
displacement must also exceed 1 mm, the largest inter-marker distance error of current optical systems (Topley &
Richards 2020). Per trial, because trials hold very different motions. The normal range already holds the trial's
soft-tissue movement, so skin-marker clusters are checked with the same rule. Each trial's fences are compared with
the other trials of its session and a trial far out of line is reported, not judged differently: where the error
fills much of a trial its fence rises and hides it; an offset that stays constant through the trial is left to the
later removal of a constant rotation. Clusters with three markers are not checkable: leaving one out leaves two,
which define no orientation and give no repair. One occurrence is one continuous run of frames with the same marker
to blame.

How we handle it: counted, and the frames are left as deposited (report only); only jumps of the reference are
masked (error 3, reference jumps or flips). The rebuild below (orientation fitted to at least three other markers,
else the frame marked invalid) is not applied in the order of correction: in skin clusters most flagged runs are
smooth deformations, the blamed marker stays within 1 mm of its own path, and the rebuild moved Camargo's flagged
frames from 0.6° to 15° off their clean neighbours (medians), Darmstadt's from 1.2° to 4.0°.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..recording import Recording
from ._result import CheckResult, entry
from .e04_gap_fill import stretches
from .e01_label_swaps import _kabsch, local_positions

ERROR = 'marker in the wrong place'
MIN_MARKERS = 4  # one left out, three to fit
# Floor of the displacement: a deviation below what current optical systems resolve is not
# counted. Topley & Richards 2020 (J Biomech 106:109820, ten passive-marker systems in gait labs, unfiltered, rotating
# bar): the largest single-frame error of an 89.9 mm inter-marker distance stayed below 1 mm (0.18-0.99 mm). It also
# removes DIODEM's sub-micrometre runs, whose markers are a template placed by the stored pose.
FLOOR_MM = 1.0
FLOOR_DEG = 1e-6  # numerical only: the influence follows from the displacement and the cluster's size
# Tukey's far-out fence: with the outlier fence (1.5 IQR) 0.1 mm of synthetic noise alone gave
# about 25 one-frame runs per 1000 frames, with 3 IQR about 0.6.
FENCE_IQR = 3.0


def trial_fence(values: np.ndarray, iqr: float | None = None) -> float:
    v = values[np.isfinite(values)]
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 + (FENCE_IQR if iqr is None else iqr) * (q3 - q1))


def _fit(local: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations R with world - centroid ≈ R (local - centroid), no reflection. local (K, 3), world
    (N, K, 3)."""
    a = local - local.mean(0)
    b = world - world.mean(1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum('nki,kj->nij', b, a))
    d = np.sign(np.linalg.det(u @ vt))
    u[:, :, 2] *= d[:, None]
    return u @ vt


def cluster_shape(m: np.ndarray, iterations: int = 3, frames: int = 3000) -> np.ndarray | None:
    """(K, 3) the cluster's marker positions relative to their centroid, from the frames with every marker present
    (at most `frames` of them, evenly spread): start from one frame, align all to it, take the median, repeat."""
    full = m[np.isfinite(m).all(axis=(1, 2))]
    if not len(full):
        return None
    full = full[np.linspace(0, len(full) - 1, min(frames, len(full))).astype(int)]
    centred = full - full.mean(1, keepdims=True)
    shape = centred[len(centred) // 2]
    for _ in range(iterations):
        r = _fit(shape, centred)
        shape = np.median(np.einsum('nji,nkj->nki', r, centred), axis=0)
    return shape


def leave_one_out(m: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame and marker (N, K): displacement (mm), influence (degrees) and the RMS residual of the other markers
    in the fit without it. NaN where the marker is missing or fewer than MIN_MARKERS are present."""
    n, k = m.shape[:2]
    disp, infl, resid = (np.full((n, k), np.nan) for _ in range(3))
    present = np.isfinite(m).all(axis=2)
    for pattern in np.unique(present[present.sum(axis=1) >= MIN_MARKERS], axis=0):
        rows = np.flatnonzero((present == pattern).all(axis=1))
        idx = np.flatnonzero(pattern)
        w = m[rows][:, idx]
        r_all = _fit(shape[idx], w)
        for j, kk in enumerate(idx):
            others = np.delete(np.arange(len(idx)), j)
            loc, wo = shape[idx][others], w[:, others]
            r = _fit(loc, wo)
            centre_w, centre_l = wo.mean(1), loc.mean(0)
            pred = np.einsum('nij,kj->nki', r, loc - centre_l) + centre_w[:, None]
            resid[rows, kk] = np.sqrt(((pred - wo) ** 2).sum(-1).mean(-1))
            pk = np.einsum('nij,j->ni', r, shape[kk] - centre_l) + centre_w
            disp[rows, kk] = np.linalg.norm(w[:, j] - pk, axis=1)
            rel = np.einsum('nji,njk->nik', r_all, r)
            infl[rows, kk] = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return disp, infl, resid


def find_misplaced(m: np.ndarray, shape: np.ndarray | None = None, iqr: float | None = None) -> dict | None:
    """Transient runs (first, frames, marker), runs reaching the trial's first or last checkable frame (not
    counted), the trial's fences and the frames checked; None if the cluster has fewer than MIN_MARKERS markers or
    no frame with all of them."""
    if m.shape[1] < MIN_MARKERS:
        return None
    shape = cluster_shape(m) if shape is None else shape
    if shape is None:
        return None
    disp, infl, resid = leave_one_out(m, shape)
    checked = np.isfinite(resid).any(axis=1)
    if not checked.any():
        return None
    fence_mm, fence_deg = trial_fence(disp, iqr), trial_fence(infl, iqr)
    limit_mm, limit_deg = max(fence_mm, FLOOR_MM), max(fence_deg, FLOOR_DEG)
    culprit = np.argmin(np.where(np.isfinite(resid), resid, np.inf), axis=1)
    rows = np.arange(len(m))
    bad = checked & (disp[rows, culprit] > limit_mm) & (infl[rows, culprit] > limit_deg)
    culprit = np.where(bad, culprit, -1)
    have = np.flatnonzero(checked)
    transient, lasting = [], []
    for k in range(m.shape[1]):
        for first, n in stretches(culprit == k):
            sl = slice(first, first + n)
            run = dict(first=first, frames=n, marker=k, displacement_mm=float(np.nanmax(disp[sl, k])),
                       influence_deg=float(np.nanmax(infl[sl, k])))
            (lasting if first <= have[0] or first + n - 1 >= have[-1] else transient).append(run)
    return dict(runs=sorted(transient, key=lambda r: r['first']), lasting=lasting, limit_mm=limit_mm,
                limit_deg=limit_deg, fence_mm=fence_mm, fence_deg=fence_deg, frames_checked=int(checked.sum()))


def check_for_misplaced_markers(recording: Recording) -> CheckResult:
    """Misplaced markers per segment, against the trial's own normal range. A segment without markers or with fewer
    than four is not checkable (detected None)."""
    per_segment, found, decided = {}, False, False
    for name, seg in recording.segments.items():
        markers = seg.reference.markers
        if not markers:
            per_segment[name] = entry(None, float('nan'), 'runs', reason='no cluster markers')
            continue
        labels = list(markers)
        res = find_misplaced(np.stack([np.asarray(v, float) for v in markers.values()], axis=1))
        if res is None:
            per_segment[name] = entry(None, float('nan'), 'runs', markers=labels,
                                      reason=f'{len(labels)} markers: fewer than {MIN_MARKERS}, not checkable'
                                      if len(labels) < MIN_MARKERS else 'no frame with every marker present')
            continue
        decided = True
        found |= bool(res['runs'])
        for r in res['runs'] + res['lasting']:
            r['label'] = labels[r['marker']]
        per_segment[name] = entry(bool(res['runs']), len(res['runs']), 'runs',
                                  frames=int(sum(r['frames'] for r in res['runs'])), markers=labels, **res)
    if not decided:
        return CheckResult(ERROR, None, 'No segment has a cluster of at least four markers; the check cannot run.',
                           per_segment)
    total = sum(p['estimate'] for p in per_segment.values() if p['detected'] is not None)
    summary = (f'{total} runs of frames with one cluster marker in the wrong place.' if found
               else 'No cluster marker leaves its place for part of a trial.')
    return CheckResult(ERROR, found, summary, per_segment)


def remove_misplaced_markers(recording: Recording, result: CheckResult, local: dict | None = None) -> Recording:
    """Copy in which the orientation of every frame with a misplaced marker is rebuilt from the other markers where at
    least three of them are present, and marked invalid otherwise.

    local: {segment: (K, 3) marker positions in the segment frame} from the participant's session; used where given,
    else the trial's own frames without a misplaced marker."""
    new, rebuilt, masked = recording.copy(), {}, {}
    for name, p in result.per_segment.items():
        if not p.get('detected'):
            continue
        ref = new.segments[name].reference
        m = np.stack([np.asarray(ref.markers[n], float) for n in ref.markers], axis=1)
        bad = np.full(len(m), -1)
        for r in p['runs']:
            bad[r['first']:r['first'] + r['frames']] = r['marker']
        cluster = (local or {}).get(name)
        if cluster is None:
            cluster = local_positions(m, ref.orientation, ref.valid & (bad < 0))
        mats = ref.orientation.as_matrix()
        lost = np.zeros(len(m), bool)
        n_rebuilt = 0
        for f in np.flatnonzero(bad >= 0):
            keep = [k for k in range(m.shape[1]) if k != bad[f] and np.isfinite(m[f, k]).all()]
            if cluster is None or len(keep) < 3:
                lost[f] = True
                continue
            mats[f] = _kabsch(cluster[keep], m[f, keep])
            n_rebuilt += 1
        ref.orientation = Rotation.from_matrix(mats)
        ref.valid = ref.valid & ~lost
        rebuilt[name], masked[name] = n_rebuilt, int(lost.sum())
    new.log_correction(ERROR, f'Rebuilt {sum(rebuilt.values())} reference frames from the cluster markers left in '
                              f'place and marked {sum(masked.values())} invalid that kept fewer than three.',
                       frames_rebuilt=rebuilt, frames_masked=masked)
    return new


# --- handler: one dataset at a time -------------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from ._report import ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import dumps, parallel_map, reader_of, truthy  # noqa: E402


def _check_trial(loader, trial) -> list[dict]:
    rec = loader.load(trial)
    session = loader.session_of(trial)
    session = f"{loader.participant_of(trial)}|{'' if session == trial else session}"
    rows = []
    for seg, p in check_for_misplaced_markers(rec).per_segment.items():
        labels = p.get('markers', [])
        run = lambda r: dict(first=r['first'], frames=r['frames'], marker=labels[r['marker']],  # noqa: E731
                             displacement_mm=r['displacement_mm'], influence_deg=r['influence_deg'])
        rows.append(dict(trial=trial, participant=str(loader.participant_of(trial)), session=session, segment=seg,
                         detected=p['detected'], estimate=p['estimate'], unit='runs', markers=len(labels),
                         frames=p.get('frames'), frames_checked=p.get('frames_checked'),
                         limit_mm=p.get('limit_mm'), limit_deg=p.get('limit_deg'), fence_mm=p.get('fence_mm'),
                         fence_deg=p.get('fence_deg'), lasting=len(p.get('lasting', [])),
                         runs=dumps([run(r) for r in p.get('runs', [])]),
                         lasting_runs=dumps([run(r) for r in p.get('lasting', [])]), reason=p.get('reason', '')))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the runs of frames with one cluster marker in the wrong place (estimate), the
    trial's fences and, in fence_mm_vs_session, its displacement fence against the median of its session (a trial
    far above hides its own error). Runs reaching the trial's first or last frame are listed in lasting_runs and not
    counted. Undecided where the segment has no cluster of at least four markers."""
    rows = []
    for out in parallel_map(loader, _check_trial, loader.trials(), processes):
        rows += [dict(out, segment='', detected=None)] if isinstance(out, dict) else out
    rows = pd.DataFrame(rows)
    if len(rows) and 'fence_mm' in rows:
        fence = pd.to_numeric(rows.fence_mm)
        rows['fence_mm_vs_session'] = fence / fence.groupby([rows.session, rows.segment]).transform('median')
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a transient run / runs."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'runs', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def correct(loader, rows: pd.DataFrame):
    """Report only: the frames are left as deposited (module description)."""
    return loader


SPEC = ErrorSpec(2, 'misplaced_markers', 'report', 'a cluster marker in the wrong place')
PARTS = {2: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_misplaced_markers = make_handler(PARTS[2])
