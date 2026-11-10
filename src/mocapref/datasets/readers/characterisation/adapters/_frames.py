"""Frame construction shared by the housing-referenced and bone-referenced human dataset adapters: a quaternion
conversion, a rigid (Procrustes) fit of a marker cluster to its own median shape, and a deterministic three-point
anatomical frame. No orientation estimator.
"""
from __future__ import annotations

import numpy as np

STANDARD_GRAVITY = 9.80665
"""m/s². Physical constant (CODATA / ISO 80000 standard gravity); used to convert accelerometers deposited
in units of g.

Units: m/s².
Provenance: physical — the defined value of ISO 80000-3 / CODATA."""

DEG = np.pi / 180.0
"""Radians per degree; used to convert gyroscopes deposited in °/s.

Units: rad per degree.
Provenance: physical — an exact unit conversion."""


def quat_to_rotmat(q: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """(..., 4) quaternions to (..., 3, 3) rotation matrices.

    ``scalar_first`` selects the (w, x, y, z) layout. The quaternions are normalised first, so a deposit
    that stores them at float32 precision does not produce a non-orthonormal matrix.
    """
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / n
    if scalar_first:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def cluster_template(markers: np.ndarray) -> np.ndarray | None:
    """The (K, 3) reference shape ``cluster_pose`` fits: the per-marker median of the centroid-removed positions
    over the frames where all markers are present; None without such a frame."""
    markers = np.asarray(markers, dtype=float)
    ok = np.isfinite(markers).all(axis=(1, 2))
    if not ok.any():
        return None
    template = np.median(markers[ok] - markers[ok].mean(axis=1)[:, None, :], axis=0)
    return template - template.mean(axis=0)


def cluster_pose(markers: np.ndarray, template: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rigid pose of a marker cluster, frame → world, by a Procrustes fit to the cluster's own median shape.

    With ``template`` (a shape from ``cluster_template`` of another trial of the same session) the frame is the
    same body-fixed frame in every trial fitted to it; without, it is this trial's own median shape, whose frame
    differs from trial to trial by the cluster's median orientation in the room.

    ``markers`` is (T, K, 3) in metres, K ≥ 3, with NaN where a marker is missing. The reference shape is the
    per-marker median position in the cluster's own centroid-removed coordinates over the frames where all
    markers are present — a measurement over the whole trial, not a fitted model, and it carries no constant
    that anyone chose. The returned frame is therefore defined only up to that arbitrary but *fixed* reference
    shape: only differences of the pose over time, and rotations relative to another frame, are meaningful.

    Returns ``(R, p)``: (T, 3, 3) rotations and (T, 3) centroids, NaN in any frame where a marker is missing.
    """
    markers = np.asarray(markers, dtype=float)
    if markers.ndim != 3 or markers.shape[1] < 3 or markers.shape[2] != 3:
        raise ValueError(f"markers must be (T, K>=3, 3), got {markers.shape}")
    t_n = markers.shape[0]
    ok = np.isfinite(markers).all(axis=(1, 2))
    r_out = np.full((t_n, 3, 3), np.nan)
    p_out = np.full((t_n, 3), np.nan)
    if not ok.any():
        return r_out, p_out

    centroid = markers.mean(axis=1)
    local = markers - centroid[:, None, :]
    if template is None:
        template = np.median(local[ok], axis=0)  # (K, 3), the trial-median cluster shape
        template = template - template.mean(axis=0)

    x = local[ok]  # (n, K, 3)
    # H = Σ_k template_k xᵀ_k, then R = U diag(1, 1, det) Vᵀ solves min ‖R template − x‖ (Kabsch/Wahba).
    h = np.einsum("nki,kj->nij", x, template)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", u, vt)))
    scale = np.ones((u.shape[0], 3))
    scale[:, 2] = d
    r = np.einsum("nij,nj,njk->nik", u, scale, vt)
    r_out[ok] = r
    p_out[ok] = centroid[ok]
    return r_out, p_out


def cluster_rigidity_deviation(markers: np.ndarray) -> np.ndarray:
    """Per frame, the largest deviation of any inter-marker distance from its trial median, in metres.

    The gate a dataset with silent marker gap-filling needs (the Geneva deposit reaches 69–311 mm here). The
    adapter *exposes* this number and never removes frames.
    """
    markers = np.asarray(markers, dtype=float)
    t_n, k = markers.shape[0], markers.shape[1]
    if k < 2:
        return np.zeros(t_n)
    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(np.linalg.norm(markers[:, i] - markers[:, j], axis=-1))
    d = np.stack(dists, axis=-1)  # (T, K(K-1)/2)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(d, axis=0)
        return np.nanmax(np.abs(d - med), axis=-1)


def frame_from_three_points(origin: np.ndarray, along: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """A right-handed orthonormal frame from three tracked points, frame → world, (T, 3, 3).

    Convention, fixed here once and used identically for every segment, subject and trial so that the
    *variation* of a frame and its rotation against another frame are comparable (data-free, no constant):

    * first axis  = unit(``along`` − ``origin``);
    * third axis  = unit(first × (``plane`` − ``origin``));
    * second axis = third × first.

    The absolute orientation of the resulting frame has no anatomical meaning — it is not a clinical
    convention and must never be reported as a joint angle. What it is for is §2.6 and §2.7 of
    ``docs/characterisation.md``: the rotation of the housing cluster against this frame, whose *constant*
    part is the sensor-to-segment alignment and whose *variation* is the mount.
    """
    origin = np.asarray(origin, dtype=float)
    e1 = np.asarray(along, dtype=float) - origin
    v = np.asarray(plane, dtype=float) - origin
    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    e3 = np.cross(e1, v)
    e3 = e3 / np.linalg.norm(e3, axis=-1, keepdims=True)
    e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=-1)  # columns are the frame axes in world coordinates
