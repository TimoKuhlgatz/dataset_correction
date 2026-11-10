"""Error 1: marker labels swapped within one segment's cluster.

What it is: two markers of a cluster carry each other's names, for a whole trial (labelled wrong at the start), from
a frame on (re-identified wrongly after an occlusion) or for a few frames. The orientation built from the cluster is
then wrong, often by close to 180°. Swaps between segments are not looked for.

How we catch it (reads only the reference). A rigid cluster keeps the distance of every
labelled marker pair. Per participant and session the reference table is the median over trials of each pair's
median distance. A trial is cut into pieces: at frames where a marker is missing (re-identification after an
occlusion) and where the markers jump onto each other's paths (from frame k-1 to k the assignment of the markers to
their positions predicted at constant velocity is cheaper with a permutation than with the stored labels). Each
piece's table is compared with the reference table; its deviation is the largest difference over the marker pairs.
Threshold: where the participant has a static trial, that trial's deviation from the reference
table (how far two correct tables differ); elsewhere the participant's typical deviation (median over its trials).
A piece above the threshold is a candidate; it is swapped if exchanging one pair of labels lowers its deviation to no
more than an ordinary correct window of the session, from the reference table and, where it exists, from the
static table (a repaired piece is only as good as a correct trial, so the candidate threshold would reject half of
them). The bound for a piece of n frames is the upper outlier fence (quartile + 1.5 IQR) of the deviations of the
session's n-frame windows. A pair whose exchange changes the reference table by less than twice that bound (a near
mirror image within the cluster) cannot be judged and is named, not counted; a piece within the bound is not a
candidate.

How we correct it: the labels are exchanged back and the orientation of the swapped frames is rebuilt from the
corrected markers: the markers' positions in the segment frame (median over the unswapped frames) are fitted to the
corrected positions (least squares, no reflection).
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.spatial.transform import Rotation

from ..recording import Recording
from ._result import CheckResult, entry

ERROR = 'marker labels swapped'


def pair_distances(m: np.ndarray) -> np.ndarray:
    """(N, P) distances of every marker pair (combinations of the K markers in order). m: (N, K, 3)."""
    return np.stack([np.linalg.norm(m[:, i] - m[:, j], axis=1) for i, j in combinations(range(m.shape[1]), 2)],
                    axis=1)


def table(m: np.ndarray) -> np.ndarray:
    """(P,) median distance of every marker pair over the frames in which all markers are present."""
    d = pair_distances(m)
    ok = np.isfinite(d).all(axis=1)
    return np.median(d[ok], axis=0) if ok.any() else np.full(d.shape[1], np.nan)


def deviation(t: np.ndarray, reference: np.ndarray) -> float:
    return float(np.nanmax(np.abs(t - reference))) if np.isfinite(t).any() else float('nan')


def jump_frames(m: np.ndarray) -> np.ndarray:
    """(N,) bool: frame k where the markers jump onto each other's paths (see the module description)."""
    n, k = m.shape[:2]
    out = np.zeros(n, bool)
    if n < 3 or k < 2:
        return out
    pred = 2 * m[1:-1] - m[:-2]  # position at frame k predicted from k-1 and k-2
    actual = m[2:]
    cost = np.linalg.norm(actual[:, None, :, :] - pred[:, :, None, :], axis=3)  # [frame, label i, position j]
    ident = np.trace(cost, axis1=1, axis2=2)
    best = ident.copy()
    for p in permutations(range(k)):
        if p != tuple(range(k)):
            best = np.minimum(best, cost[:, np.arange(k), list(p)].sum(axis=1))
    ok = np.isfinite(ident)
    out[2:] = ok & (best < ident)
    return out


def pieces(m: np.ndarray) -> list[tuple[int, int]]:
    """(first frame, frames) of the runs with every marker present, cut at jump frames."""
    present = np.isfinite(m).all(axis=(1, 2))
    cut = jump_frames(m)
    runs, start = [], None
    for i in range(len(m) + 1):
        if i < len(m) and present[i] and not (cut[i] and start is not None):
            if start is None:
                start = i
            continue
        if start is not None:
            runs.append((start, i - start))
            start = None
        if i < len(m) and present[i]:
            start = i  # a jump frame starts the next piece
    return runs


def best_exchange(t: np.ndarray, k: int, reference: np.ndarray, static: np.ndarray | None):
    """The pair (i, j) whose exchange fits best, and the deviations it reaches from the reference and static table."""
    pairs = list(combinations(range(k), 2))
    best = None
    for a, b in pairs:
        order = list(range(k))
        order[a], order[b] = b, a
        swapped = np.array([t[pairs.index(tuple(sorted((order[i], order[j]))))] for i, j in pairs])
        dev = deviation(swapped, reference)
        dev_static = deviation(swapped, static) if static is not None else float('nan')
        if best is None or dev < best[1]:
            best = ((a, b), dev, dev_static)
    return best


def exchange_indices(k: int) -> np.ndarray:
    """(E, P): for each exchange of one pair of the k labels, where each pair's distance comes from."""
    pairs = list(combinations(range(k), 2))
    out = []
    for a, b in pairs:
        order = list(range(k))
        order[a], order[b] = b, a
        out.append([pairs.index(tuple(sorted((order[i], order[j])))) for i, j in pairs])
    return np.array(out)


def best_labelled_deviation(t: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Deviation of each table in t (..., P) in its best labelling: its own or with one pair exchanged."""
    t = np.asarray(t, float)
    own = np.max(np.abs(t - reference), axis=-1)
    swapped = np.max(np.abs(t[..., exchange_indices(k)] - reference), axis=-1).min(axis=-1)
    return np.minimum(own, swapped)


def fence(values) -> float:
    """Tukey's upper outlier fence: upper quartile + 1.5 x interquartile range."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float('nan')
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


class SessionReference:
    """The reference of one segment over one participant's session.

    reference: median over the trials of their tables. static: median table of the static trials, or None.
    threshold (candidates): the static trials' deviation from the reference where they exist, else the trials'
    typical (median) deviation. bound(n) (confirmation): the upper outlier fence (quartile + 1.5 IQR) of the
    deviations of the session's n-frame windows, each in its best labelling (own labels or one pair exchanged);
    a repaired piece must look like an ordinary correct window (with the worst window instead, one damaged stretch
    set the limit for the whole session). Without a window of that length, the fence of the whole trials' deviations.
    checkable(n): the exchanges that change the reference table by more than twice bound(n); a pair that is nearly
    a mirror image within the cluster (Trento's near-equilateral 20 mm triangles) cannot be judged from distances
    and is left out.
    """

    def __init__(self, clusters: list[np.ndarray], static_clusters: list[np.ndarray] = ()):
        self.k = clusters[0].shape[1]
        self.clusters = [c[np.isfinite(c).all(axis=(1, 2))] for c in clusters]
        tables = [table(c) for c in clusters]
        self.reference = np.nanmedian(np.stack(tables), axis=0)
        statics = [table(c) for c in static_clusters]
        if statics:
            self.static = np.nanmedian(np.stack(statics), axis=0)
            self.threshold = max(deviation(t, self.reference) for t in statics)
            self.source = 'static trial'
        else:
            self.static = None
            self.threshold = float(np.nanmedian([deviation(t, self.reference) for t in tables]))
            self.source = 'typical trial'
        self.whole = fence(best_labelled_deviation(np.stack(tables), self.reference, self.k))
        idx = exchange_indices(self.k)
        self.signal = np.max(np.abs(self.reference[idx] - self.reference), axis=1)  # per exchange
        self.pairs = list(combinations(range(self.k), 2))
        self._bounds = {}

    def bound(self, n: int) -> float:
        if n not in self._bounds:
            devs = []
            for c in self.clusters:
                if len(c) < 2 * n:
                    continue
                d = pair_distances(c)
                w = d[:len(d) // n * n].reshape(-1, n, d.shape[1])
                devs.append(best_labelled_deviation(np.median(w, axis=1), self.reference, self.k))
            self._bounds[n] = fence(np.concatenate(devs)) if devs else self.whole
        return self._bounds[n]

    def checkable(self, n: int) -> np.ndarray:
        """(E,) bool: exchanges that move the reference table by more than twice bound(n): only then can a correct
        piece (within the bound of the reference) not be mistaken for a swapped one (within the bound of the
        exchanged reference)."""
        return self.signal > 2 * self.bound(n)


def find_swaps(m: np.ndarray, ref: SessionReference) -> list[dict]:
    """Swapped pieces of one segment: first frame, frames, exchanged pair, deviation before and after. A piece is a
    candidate above ref.threshold and ref.bound(n) (it does not look like an ordinary correct window); it is swapped if exchanging one checkable pair lowers its deviation to within
    ref.bound(n), from the reference table and, where present, from the static table."""
    out = []
    idx = exchange_indices(m.shape[1])
    for first, n in pieces(m):
        t = table(m[first:first + n])
        dev = deviation(t, ref.reference)
        if not np.isfinite(dev) or dev <= max(ref.threshold, ref.bound(n)):
            continue  # looks like an ordinary correct window
        ok = ref.checkable(n)
        if not ok.any():
            continue
        after = np.max(np.abs(t[idx] - ref.reference), axis=1)
        after[~ok] = np.inf
        e = int(np.argmin(after))
        dev_after = float(after[e])
        dev_static = deviation(t[idx[e]], ref.static) if ref.static is not None else float('nan')
        limit = ref.bound(n)
        if dev_after < dev and dev_after <= limit and (ref.static is None or dev_static <= limit):
            out.append(dict(first=first, frames=n, pair=ref.pairs[e], deviation=dev, deviation_after=dev_after,
                            limit=limit))
    return out


def check_for_label_swaps(recording: Recording, references: dict) -> CheckResult:
    """Swapped labels per segment. references: {segment: SessionReference} over the participant's session; a segment
    without markers or reference is undecided."""
    per_segment, found, decided = {}, False, False
    for name, seg in recording.segments.items():
        markers = seg.reference.markers
        if not markers or name not in references or len(markers) < 3:
            per_segment[name] = entry(None, float('nan'), 'pieces', reason='no cluster markers or reference table')
            continue
        m = np.stack([np.asarray(v, float) for v in markers.values()], axis=1)
        swaps = find_swaps(m, references[name])
        decided = True
        found |= bool(swaps)
        per_segment[name] = entry(bool(swaps), len(swaps), 'pieces', frames=int(sum(s['frames'] for s in swaps)),
                                  swaps=swaps, threshold_mm=references[name].threshold,
                                  not_checkable=[tuple(list(markers)[i] for i in references[name].pairs[e])
                                                 for e in np.flatnonzero(~references[name].checkable(1))],
                                  threshold_from=references[name].source, markers=list(markers))
    if not decided:
        return CheckResult(ERROR, None, 'No segment has cluster markers and a reference table.', per_segment)
    total = sum(p['estimate'] for p in per_segment.values() if p['detected'] is not None)
    summary = (f'{total} pieces with two labels swapped within a cluster.' if found
               else 'No piece of any cluster fits better with two labels exchanged.')
    return CheckResult(ERROR, found, summary, per_segment)


def local_positions(m: np.ndarray, orientation: Rotation, valid: np.ndarray) -> np.ndarray | None:
    """(K, 3) the markers' positions in the segment frame relative to their centroid, median over the frames with a
    valid reference and every marker present; None without such a frame."""
    good = np.asarray(valid, bool) & np.isfinite(m).all(axis=(1, 2))
    if not good.any():
        return None
    mats = orientation.as_matrix()[good]
    centred = m[good] - m[good].mean(axis=1, keepdims=True)
    return np.median(np.einsum('nji,nkj->nki', mats, centred), axis=0)


def _kabsch(local: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Rotation R (no reflection) with world - centroid ≈ R (local - centroid). local, world: (K, 3)."""
    a, b = local - local.mean(0), world - world.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


def remove_label_swaps(recording: Recording, result: CheckResult, local: dict | None = None) -> Recording:
    """Copy with the swapped labels exchanged back and the orientation of those frames rebuilt from the markers.

    local: {segment: (K, 3) marker positions in the segment frame} from the participant's session (local_positions
    over its unswapped trials); used where given, else the trial's own unswapped frames. A segment with neither
    keeps its labels and has the swapped frames masked."""
    new, fixed = recording.copy(), {}
    for name, p in result.per_segment.items():
        if not p.get('detected'):
            continue
        ref = new.segments[name].reference
        labels = list(ref.markers)
        m = np.stack([np.asarray(ref.markers[n], float) for n in labels], axis=1)
        swapped = np.zeros(len(m), bool)
        for s in p['swaps']:
            a, b = s['pair']
            sl = slice(s['first'], s['first'] + s['frames'])
            m[sl, [a, b]] = m[sl, [b, a]]
            swapped[sl] = True
        cluster = (local or {}).get(name)
        if cluster is None:
            cluster = local_positions(m, ref.orientation, ref.valid & ~swapped)
        if cluster is None:
            ref.valid = ref.valid & ~swapped  # nothing to learn the cluster from: mask instead
            fixed[name] = 0
            continue
        mats = ref.orientation.as_matrix()
        for k in np.flatnonzero(swapped & np.isfinite(m).all(axis=(1, 2))):
            mats[k] = _kabsch(cluster, m[k])
        ref.orientation = Rotation.from_matrix(mats)
        for i, n in enumerate(labels):
            ref.markers[n] = m[:, i]
        fixed[name] = int(swapped.sum())
    new.log_correction(ERROR, f'Exchanged swapped labels back in {sum(fixed.values())} frames and rebuilt their '
                              'orientation from the corrected markers.', frames_fixed=fixed)
    return new


# --- handler: one dataset at a time -------------------------------------------------------------------------------

import re  # noqa: E402
from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from ._report import CorrectedLoader, ErrorSpec, count_trials, make_handler  # noqa: E402
from ._rows import by_participant, dumps, loads, parallel_map, reader_of, truthy  # noqa: E402

STATIC = re.compile(r'n-pose|npose|tpose|static', re.I)
"""Trial names of static trials, whose tables set the candidate threshold of their session."""


def _cluster_markers(rec: Recording) -> dict:
    return {s: np.stack([np.asarray(v, float) for v in seg.reference.markers.values()], axis=1)
            for s, seg in rec.segments.items() if seg.reference.markers and len(seg.reference.markers) >= 3}


def _check_session(loader, group) -> list[dict]:
    """Rows of every trial and segment of one participant's session (SessionReference over its trials)."""
    key, trials = group
    clusters, recs = {}, {}
    for t in trials:
        rec = loader.load(t)
        if not rec.segments:
            continue
        recs[t] = rec
        for s, m in _cluster_markers(rec).items():
            clusters.setdefault(s, {})[t] = m
    refs = {}
    for s, per_trial in clusters.items():
        if len({m.shape[1] for m in per_trial.values()}) == 1:  # one cluster layout in the whole session
            refs[s] = SessionReference(list(per_trial.values()),
                                       [m for t, m in per_trial.items() if STATIC.search(t)])
    rows = []
    for t, rec in recs.items():
        for s, p in check_for_label_swaps(rec, refs).per_segment.items():
            labels = p.get('markers', [])
            swaps = [dict(first=w['first'], frames=w['frames'], pair=[labels[i] for i in w['pair']],
                          deviation_mm=w['deviation'], deviation_after_mm=w['deviation_after'], limit_mm=w['limit'])
                     for w in p.get('swaps', [])]
            rows.append(dict(trial=t, participant=str(loader.participant_of(t)), session=key, segment=s,
                             detected=p['detected'], estimate=p['estimate'], unit='pieces',
                             frames=p.get('frames', 0), reference_frames=len(rec.segments[s].reference),
                             swaps=dumps(swaps), threshold_mm=p.get('threshold_mm'),
                             threshold_from=p.get('threshold_from'),
                             not_checkable=';'.join('-'.join(x) for x in p.get('not_checkable', [])),
                             reason=p.get('reason', '')))
    return rows


def check_dataset(loader, dataset: str, processes: int = 1, **options) -> pd.DataFrame:
    """One row per trial and segment: the pieces with two labels of the cluster swapped (estimate), each in `swaps`
    (first frame, frames, the exchanged pair). Undecided where the segment has no cluster of three markers or its
    session no common cluster layout. The reference tables are pooled per participant and session."""
    rows = []
    for out in parallel_map(loader, _check_session, by_participant(loader, session=True), processes):
        if isinstance(out, dict):  # the session could not be read
            rows += [dict(trial=t, participant=str(loader.participant_of(t)), segment='', detected=None,
                          error=out['error']) for t in out['trial'][1]]
        else:
            rows += out
    rows = pd.DataFrame(rows)
    rows['reader'] = reader_of(loader)
    return rows


def prevalence(rows: pd.DataFrame) -> dict:
    """Trials with a swapped piece / swapped pieces."""
    hit = rows.detected.map(truthy)
    return count_trials(rows, 'pieces', int(pd.to_numeric(rows.estimate[hit]).sum()) if hit.any() else 0)


def _repair(rec: Recording, trial, *, swaps: dict, base) -> Recording:
    """remove_label_swaps with the pieces of `swaps` ({trial: {segment: (reference frames, [swap])}}); a whole-trial
    swap takes the markers' positions in the segment frame from another trial of the participant."""
    found = swaps.get(str(trial))
    if not found:
        return rec
    per_segment, local = {}, {}
    for seg, (n_frames, pieces_) in found.items():
        if seg not in rec.segments or len(rec.segments[seg].reference) != n_frames:
            continue  # another reader's reference (e.g. a re-paired trial): the frames do not match
        labels = list(rec.segments[seg].reference.markers)
        per_segment[seg] = dict(detected=True, swaps=[dict(first=w['first'], frames=w['frames'],
                                                           pair=tuple(labels.index(x) for x in w['pair']))
                                                      for w in pieces_])
        if sum(w['frames'] for w in pieces_) >= n_frames:
            others = [t for t in base.trials() if str(t) != str(trial) and
                      base.participant_of(t) == base.participant_of(trial) and seg not in swaps.get(str(t), {})]
            for t in others:
                other = base.load(t).segments.get(seg)
                if other is not None and other.reference.markers and list(other.reference.markers) == labels:
                    m = np.stack([np.asarray(v, float) for v in other.reference.markers.values()], axis=1)
                    local[seg] = local_positions(m, other.reference.orientation, other.reference.valid)
                    break
    if not per_segment:
        return rec
    return remove_label_swaps(rec, CheckResult(ERROR, True, '', per_segment), local)


def correct(loader, rows: pd.DataFrame):
    """The swapped labels exchanged back and those frames' orientation rebuilt from the markers."""
    swaps = {}
    for r in rows[rows.detected.map(truthy)].itertuples():
        swaps.setdefault(str(r.trial), {})[r.segment] = (int(r.reference_frames), loads(r.swaps, []))
    if not swaps:
        return loader
    return CorrectedLoader(loader, partial(_repair, swaps=swaps, base=loader), 'marker labels swapped')


SPEC = ErrorSpec(1, 'label_swaps', 'fix', 'marker labels swapped within a cluster')
PARTS = {1: dict(spec=SPEC, check_dataset=check_dataset, prevalence=prevalence, correct=correct)}
handle_label_swaps = make_handler(PARTS[1])
