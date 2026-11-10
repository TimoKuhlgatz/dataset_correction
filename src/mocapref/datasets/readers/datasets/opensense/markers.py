"""The OpenSense marker file: reading it, grouping its markers into the eight sensor plates, and fitting a
rigid frame to each plate frame by frame.

**What a plate is.** Every Xsens housing in this deposit carries a small machined plate with four (sometimes
five) reflective markers on it, named by the depositors after the body the sensor sits on and after the role
the marker plays in the plate's own frame — ``R.Femur_IMU_O`` is the origin of the right-thigh plate,
``_X`` and ``_Y`` span its first two axes, ``_D`` and ``_5`` are extra markers that only make the least-squares
fit better conditioned. Because the plate is machined around the housing, its orientation *is* the sensor
housing's orientation up to one constant rotation, and that is what makes this dataset's reference a housing
reference rather than a skin one.

**What this module does not do.** It never reads the shipped ``markerplate_orientations.sto``. That file is
defective — its left and right foot columns are permutations of one another — which is the whole reason this
reconstruction exists. It also uses no model, no inverse kinematics and no fusion: every number below comes
out of the marker coordinates by linear algebra alone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MM_PER_M = 1e3
"""Millimetres in a metre — written once, because the plate diagnostics are reported in millimetres while
every coordinate in this module is in metres.

Units: millimetres per metre.
Provenance: physical — the definition of the millimetre."""

UNIT_SCALE_M = {"mm": 1e-3, "m": 1.0}
"""Marker-coordinate unit named in the marker file's header -> the factor that turns it into metres.

Units: metres per deposited unit.
Provenance: physical — the definition of the millimetre; the header names which of the two the file uses."""

HEADER_KEY_LINE = 1
"""Line of the marker file holding the names of the numeric header fields (``DataRate``, ``Units``, ...).

Units: line index, zero-based.
Provenance: read-from-dataset — the Motion Analysis / OpenSim ``.trc`` layout as the deposit writes it."""

HEADER_VALUE_LINE = 2
"""Line holding the values of those header fields.

Units: line index, zero-based.
Provenance: read-from-dataset — the same layout."""

MARKER_NAME_LINE = 3
"""Line holding the marker names, one name every three columns after ``Frame#`` and ``Time``.

Units: line index, zero-based.
Provenance: read-from-dataset — the same layout."""

FIRST_DATA_LINE = 5
"""First line of numeric data (line 4 holds the per-coordinate sub-header ``X1 Y1 Z1 ...``).

Units: line index, zero-based.
Provenance: read-from-dataset — the same layout."""

TIME_COLUMN = 1
"""Column of the time vector; column 0 is the frame number.

Units: column index, zero-based.
Provenance: read-from-dataset — the same layout."""

COORDINATES_PER_MARKER = 3
"""Columns one marker occupies in the data block: its x, y and z.

Units: columns per marker.
Provenance: read-from-dataset — the same layout; a marker trajectory is a three-vector."""

PLATE_PREFIX = {
    "Back": "torso_imu",
    "Pelvis": "pelvis_imu",
    "R.Femur": "femur_r_imu",
    "R.Tibia": "tibia_r_imu",
    "R.Foot": "calcn_r_imu",
    "L.Femur": "femur_l_imu",
    "L.Tibia": "tibia_l_imu",
    "L.Foot": "calcn_l_imu",
}
"""Marker-name prefix in the marker file -> the sensor name the deposit's own sensor mapping uses. Eight
plates: trunk, pelvis, and thigh, shank and foot on each side."""

MARKER_SUFFIXES = ("O", "D", "X", "Y", "5")
"""The marker roles tried on every plate, in this order. ``O``, ``X`` and ``Y`` define the plate's nominal
frame and must all be present; ``D`` and ``5`` are extra cluster markers that enter the least-squares fit
only."""

FRAME_MARKERS = ("O", "X", "Y")
"""The three markers a plate must carry for its nominal frame to be definable."""

MIN_PLATE_MARKERS = 3
"""Fewest markers a plate must have in the file at all before it is considered present.

Units: markers.
Provenance: data-free — three non-collinear points is the smallest set that fixes a rigid frame."""

ORIGIN_GAP_TOLERANCE_M = 1e-6
"""A marker whose coordinates sit this close to the laboratory origin is treated as a gap: some exporters
write an exact ``0, 0, 0`` for a dropped marker rather than a blank field.

Units: metres.
Provenance: data-free — a test for the exact zero the exporter writes, at a distance far below any real marker
position in a laboratory whose origin is on the floor."""

RIGIDITY_MIN_FRAMES = 10
"""Fewest frames in which both markers of a pair must be visible before the pair's distance statistics are
formed at all.

Units: frames.
Provenance: data-free — a minimum sample count; below it the statistic is not formed and nothing is reported.
"""

RIGIDITY_REPORTED_PERCENTILE = 95
"""The percentile of the absolute inter-marker distance deviation the diagnostics report beside its spread and
its maximum. Nothing is selected on it; it is one of the numbers written into the trial's diagnostics.

Units: percent.
Provenance: data-free — a reported percentile of the whole distribution, not a threshold."""

RIGIDITY_DEVIATION_M = 5e-3
"""The inter-marker distance deviation above which a frame counts towards ``frac_gt5mm`` in the diagnostics —
the fraction of frames in which the plate does not behave as a rigid body.

Units: metres.
Provenance: inherited-protocol — a fixed protocol value of the reconstruction, not tuned on this dataset."""


def read_marker_file(path: str | Path) -> dict:
    """One ``.trc`` -> ``{'rate', 'time' (T,), 'markers': {name: (T, 3) metres}, ...}``, gaps as ``NaN``.

    Layout (Motion Analysis / OpenSim, as the deposit writes it): a numeric header on two lines, the marker
    names one every three columns, a per-coordinate sub-header, then the data. Units come out of the header
    and are converted to metres here, so nothing downstream carries a unit choice.
    """
    lines = Path(path).read_text(errors="replace").split("\n")
    header_keys = lines[HEADER_KEY_LINE].split("\t")
    header_values = lines[HEADER_VALUE_LINE].split("\t")
    header = dict(zip([k.strip() for k in header_keys], [v.strip() for v in header_values]))
    rate = float(header["DataRate"])
    units = header.get("Units", "mm").strip().lower()
    scale = UNIT_SCALE_M[units]

    names_row = lines[MARKER_NAME_LINE].split("\t")
    # names_row = ['Frame#', 'Time', name_1, '', '', name_2, '', '', ...]
    names = [n.strip() for n in names_row[TIME_COLUMN + 1:]]
    marker_names = [names[i] for i in range(0, len(names), 3) if names[i]]

    rows = []
    for line in lines[FIRST_DATA_LINE:]:
        if not line.strip():
            continue
        rows.append(line.split("\t"))
    n_columns = TIME_COLUMN + 1 + COORDINATES_PER_MARKER * len(marker_names)
    data = np.full((len(rows), n_columns), np.nan)
    for i, row in enumerate(rows):
        for j in range(min(n_columns, len(row))):
            text = row[j].strip()
            if text:
                data[i, j] = float(text)
    time = data[:, TIME_COLUMN]
    markers = {name: data[:, 2 + 3 * k:5 + 3 * k] * scale for k, name in enumerate(marker_names)}
    return {"rate": rate, "time": time, "markers": markers, "n_frames": len(rows),
            "header": header, "path": str(path)}


def plate_markers(marker_file: dict) -> dict[str, dict[str, np.ndarray]]:
    """``{sensor name: {marker role: (T, 3)}}`` for every plate fully present in the file."""
    out = {}
    for prefix, sensor in PLATE_PREFIX.items():
        got = {}
        for suffix in MARKER_SUFFIXES:
            key = f"{prefix}_IMU_{suffix}"
            if key in marker_file["markers"]:
                got[suffix] = marker_file["markers"][key]
        if all(s in got for s in FRAME_MARKERS) and len(got) >= MIN_PLATE_MARKERS:
            out[sensor] = got
    return out


def visibility(cluster: dict[str, np.ndarray]) -> dict:
    """Which markers of one plate are visible in which frames.

    A marker counts as missing when any of its three coordinates is ``NaN`` **or** when the point sits exactly
    at the laboratory origin, which is how some exporters write a dropped marker.
    """
    names = sorted(cluster)
    seen = {}
    for name in names:
        track = cluster[name]
        seen[name] = np.isfinite(track).all(axis=1) & (np.linalg.norm(track, axis=1) > ORIGIN_GAP_TOLERANCE_M)
    stacked = np.stack([seen[n] for n in names], axis=1)  # (T, K)
    return {"names": names, "vis": stacked, "n_visible": stacked.sum(axis=1)}


def rigidity(cluster: dict[str, np.ndarray], vis: np.ndarray, names: list[str]) -> dict:
    """Inter-marker distance statistics of one plate — the check that it really is a rigid body.

    A rigid plate has constant pairwise distances. Returned per pair: the median distance, the spread of the
    deviation from it, the reported percentile and maximum of that deviation, and the fraction of frames in
    which the pair is off by more than :data:`RIGIDITY_DEVIATION_M` (the cluster deforms, or a marker is
    mislabelled). ``worst_dev_m`` is the per-frame worst deviation over all pairs, and is what the reference
    frame of the Kabsch fit is chosen by.
    """
    n_markers = len(names)
    pairs, per_frame = [], []
    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            first, second = cluster[names[i]], cluster[names[j]]
            both = vis[:, i] & vis[:, j]
            if both.sum() < RIGIDITY_MIN_FRAMES:
                continue
            distance = np.linalg.norm(first[both] - second[both], axis=1)
            median = float(np.median(distance))
            deviation = distance - median
            pairs.append({
                "pair": f"{names[i]}-{names[j]}",
                "median_mm": median * MM_PER_M,
                "sd_mm": float(np.std(deviation)) * MM_PER_M,
                "p95_abs_mm": float(np.percentile(np.abs(deviation), RIGIDITY_REPORTED_PERCENTILE)) * MM_PER_M,
                "max_abs_mm": float(np.max(np.abs(deviation))) * MM_PER_M,
                "frac_gt5mm": float(np.mean(np.abs(deviation) > RIGIDITY_DEVIATION_M)),
                "n": int(both.sum()),
            })
            filled = np.full(len(first), np.nan)
            filled[both] = np.abs(deviation)
            per_frame.append(filled)
    stacked = np.stack(per_frame, axis=1) if per_frame else np.zeros((len(vis), 0))
    with np.errstate(invalid="ignore"):
        worst = np.nanmax(stacked, axis=1) if stacked.shape[1] else np.zeros(len(vis))
    return {"pairs": pairs, "worst_dev_m": worst}


def nominal_frame(origin: np.ndarray, x_marker: np.ndarray, y_marker: np.ndarray) -> np.ndarray:
    """``(T, 3, 3)`` rotation laboratory<-plate built from the plate's own O / X / Y markers.

    The depositors named these markers origin, x-axis and y-axis, so this is the plate's own frame, defined
    without any model, any IMU and any fusion. The returned matrices hold the plate's axes as **columns**.
    """
    ex = x_marker - origin
    ex = ex / np.linalg.norm(ex, axis=1, keepdims=True)
    toward_y = y_marker - origin
    ez = np.cross(ex, toward_y)
    ez = ez / np.linalg.norm(ez, axis=1, keepdims=True)
    ey = np.cross(ez, ex)
    return np.stack([ex, ey, ez], axis=2)


def kabsch_series(cluster: dict[str, np.ndarray], vis: np.ndarray, names: list[str], local: np.ndarray,
                  min_markers: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame rigid fit of the reference cluster ``local`` onto the observed markers.

    Returns ``(R (T, 3, 3), rmse (T,))``; frames with fewer than ``min_markers`` visible markers get ``NaN``.
    Textbook Kabsch/Horn with the reflection guard, vectorised over the frames that share one visibility
    pattern — which is what makes a ten-minute trial a second of arithmetic rather than a minute of it.
    """
    n_frames = vis.shape[0]
    observed = np.stack([cluster[n] for n in names], axis=1)  # (T, K, 3) laboratory frame
    rotation = np.full((n_frames, 3, 3), np.nan)
    rmse = np.full(n_frames, np.nan)
    visible = vis.astype(float)
    codes = (visible * (2 ** np.arange(len(names)))).sum(axis=1).astype(np.int64)
    for code in np.unique(codes):
        selected = codes == code
        index = np.where(np.array([(code >> k) & 1 for k in range(len(names))], bool))[0]
        if len(index) < min_markers:
            continue
        reference = local[index]  # (k, 3)
        points = observed[selected][:, index, :]  # (m, k, 3)
        if not np.isfinite(points).all():
            continue
        centred_points = points - points.mean(axis=1, keepdims=True)
        centred_reference = reference - reference.mean(axis=0, keepdims=True)
        # H = sum_k P_k Q_k^T ; SVD H = U S V^T ; R = U diag(1, 1, det) V^T
        covariance = np.einsum("mki,kj->mij", centred_points, centred_reference)
        left, _singular, right = np.linalg.svd(covariance)
        determinant = np.sign(np.linalg.det(np.einsum("mij,mjk->mik", left, right)))
        guard = np.zeros((len(left), 3, 3))
        guard[:, 0, 0] = guard[:, 1, 1] = 1.0
        guard[:, 2, 2] = determinant
        fitted = np.einsum("mij,mjk,mkl->mil", left, guard, right)  # laboratory <- plate
        residual = centred_points - np.einsum("mij,kj->mki", fitted, centred_reference)
        rotation[selected] = fitted
        rmse[selected] = np.sqrt((residual ** 2).sum(axis=2).mean(axis=1))
    return rotation, rmse


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """``(T, 3, 3)`` -> ``(T, 4)`` scalar-first quaternions, ``NaN`` rows preserved."""
    from scipy.spatial.transform import Rotation

    out = np.full((len(rotation), 4), np.nan)
    finite = np.isfinite(rotation).all(axis=(1, 2))
    if finite.any():
        out[finite] = Rotation.from_matrix(rotation[finite]).as_quat()[:, [3, 0, 1, 2]]
    return out
