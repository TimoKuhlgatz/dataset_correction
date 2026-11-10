"""Primitives shared by the simulation-arm and rigid-chain adapters.

Nothing here estimates orientation (specification rule 2 of ``docs/characterisation.md``). :func:`cluster_frame`
fits **one constant** marker configuration by least squares over a whole trial and then solves a per-frame
orthogonal Procrustes problem — a rigid registration of a known template, which is a measurement, not a filter.
Its per-frame residual is returned alongside, because a large residual makes the frame meaningless.
"""
from __future__ import annotations

import numpy as np

DEG = np.pi / 180.0
"""Units: rad per degree.
Provenance: physical — an exact unit conversion."""
G_TO_M_S2 = 9.80665
"""Standard gravity, for deposits that publish the accelerometer in g.

Units: m/s².
Provenance: physical — the defined value (CIPM 1901 / ISO 80000-3)."""


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first (``w, x, y, z``) unit quaternions → (..., 3, 3) rotation matrices.

    The matrix maps a vector in the quaternion's *body* frame to the world frame, i.e. the convention
    :class:`~mocapref.datasets.readers.characterisation.schema.Pose` asks for (``rotate(q, v_body) = v_world``). Every adapter
    that uses this states, in its own docstring, how it established the deposit's convention.
    """
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.divide(q, n, out=np.full_like(q, np.nan), where=n > 0)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def rotation_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T,) radians, the angle of ``aᵀb`` — used only to report how far two deposited frames disagree."""
    m = np.einsum("tji,tjk->tik", a, b)
    trace = np.trace(m, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def cluster_frame(markers: np.ndarray, r_at_first_frame: np.ndarray | None = None,
                  rigid_tolerance_m: float | None = None):
    """A rigid frame from a marker cluster, with its own residual.

    ``markers`` is ``(T, K, 3)`` in metres, ``K ≥ 3``. The template is the cluster's configuration at the first
    fully-observed frame, expressed in ``r_at_first_frame`` when one is given — ``(3, 3)`` for that frame, or
    ``(T, 3, 3)`` from which that frame's is taken — (so the returned rotation is on
    the same constant as the deposit's own frame; without it the frame is identity at that first frame, which
    fixes the constant arbitrarily — every measure that uses the frame must then be invariant to a constant
    rotation, and says so).

    Returns ``(R, p, residual_m)``: ``R`` ``(T, 3, 3)`` template → world, ``p`` ``(T, 3)`` the cluster centroid
    in world coordinates, ``residual_m`` ``(T,)`` the root-mean-square distance between the rotated template and
    the observed markers — the cluster's own rigidity residual, in metres. Frames with any missing marker come
    back as ``NaN``.

    With ``rigid_tolerance_m`` the cluster's own geometry judges every frame: an inter-marker distance further
    than the tolerance from its median over the recording marks the frame as not rigid. Where exactly one largest
    marker subset (at least three) keeps all its distances, the marker at fault is identified and the frame is
    registered on that subset against the same template, so the rigid body's frame is unchanged; where several
    subsets hold the fault is ambiguous and small, and the registration on all markers stands (measured on the
    cadaver knee with the gyroscope as judge: the subset wins 75–98 % of identified frames and 50 % of ambiguous
    ones); where none holds the frame comes back as ``NaN``. The template is then the first frame that is rigid, not
    merely observed. Without the argument nothing changes.
    """
    markers = np.asarray(markers, float)
    t_len = markers.shape[0]
    good = ~np.isnan(markers).any(axis=(1, 2))
    if not good.any():
        nan_r = np.full((t_len, 3, 3), np.nan)
        return nan_r, np.full((t_len, 3), np.nan), np.full(t_len, np.nan)
    first = int(np.argmax(good))
    if rigid_tolerance_m is not None:
        rigid = good & rigid_frames(markers, rigid_tolerance_m)
        if rigid.any():
            first = int(np.argmax(rigid))
    centroid = markers.mean(axis=1)
    centred = markers - centroid[:, None, :]
    template = centred[first]  # (K, 3) in world coordinates at the first good frame
    if r_at_first_frame is not None:
        r_first = np.asarray(r_at_first_frame, float)
        if r_first.ndim == 3:  # the deposit's frame at every sample: take it where the template was taken
            r_first = r_first[first]
        template = template @ r_first  # express it in the deposit's frame

    h = np.einsum("ki,tkj->tij", template, centred)  # (T, 3, 3)
    h = np.where(good[:, None, None], h, np.eye(3))
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", vt.transpose(0, 2, 1), u.transpose(0, 2, 1))))
    correction = np.zeros((t_len, 3, 3))
    correction[:, 0, 0] = correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = d
    rot = np.einsum("tij,tjk,tkl->til", vt.transpose(0, 2, 1), correction, u.transpose(0, 2, 1))

    fitted = np.einsum("tij,kj->tki", rot, template)
    residual = np.sqrt(((fitted - centred) ** 2).sum(axis=-1).mean(axis=-1))
    rot[~good] = np.nan
    centroid[~good] = np.nan
    residual[~good] = np.nan
    if rigid_tolerance_m is not None:
        rot, residual = _register_on_rigid_subsets(markers, centred, template, rot, residual, good,
                                                   rigid_tolerance_m)
    return rot, centroid, residual


def rigid_frames(markers: np.ndarray, tolerance_m: float) -> np.ndarray:
    """``(T,)`` True where every inter-marker distance is within ``tolerance_m`` of its median over the recording."""
    from itertools import combinations

    pairs = list(combinations(range(markers.shape[1]), 2))
    dist = np.stack([np.linalg.norm(markers[:, i] - markers[:, j], axis=1) for i, j in pairs], axis=1)
    return (np.abs(dist - np.nanmedian(dist, axis=0)) <= tolerance_m).all(axis=1)


def _register_on_rigid_subsets(markers, centred, template, rot, residual, good, tolerance_m):
    """Frames that are not rigid: registered on the one marker subset that holds, kept when several hold, ``NaN``
    when none does."""
    from itertools import combinations

    k = markers.shape[1]
    pairs = list(combinations(range(k), 2))
    dist = np.stack([np.linalg.norm(markers[:, i] - markers[:, j], axis=1) for i, j in pairs], axis=1)
    off = np.abs(dist - np.nanmedian(dist, axis=0)) > tolerance_m
    rot, residual = rot.copy(), residual.copy()
    for t in np.flatnonzero(good & off.any(axis=1)):
        holding = []
        for size in range(k - 1, 2, -1):
            holding = [list(subset) for subset in combinations(range(k), size)
                       if not off[t, [n for n, (i, j) in enumerate(pairs) if i in subset and j in subset]].any()]
            if holding:
                break
        if not holding:
            rot[t], residual[t] = np.nan, np.nan
            continue
        if len(holding) > 1:
            continue  # which marker is at fault is ambiguous: the registration on all markers stands
        best = (0.0, holding[0])
        idx = best[1]
        a = template[idx] - template[idx].mean(axis=0)
        b = markers[t, idx] - markers[t, idx].mean(axis=0)
        u, _, vt = np.linalg.svd(b.T @ a)
        r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        rot[t] = r
        residual[t] = np.sqrt((((a @ r.T) - b) ** 2).sum(axis=-1).mean())
    return rot, residual
