"""Camargo et al. 2021: lower-limb biomechanics on a treadmill, ramps, stairs and level ground (Georgia Tech), part 1
(subjects AB06-AB14). Reader name ``camargo-treadmill``.

Files read (``<subject>/<date>/<mode>/...``, from ``_extracted/`` where present, else straight from the subject
archive ``<subject>.zip``, since the local extraction is partial):
  ``imu/<trial>.mat``      four six-axis units, 200 Hz
  ``markers/<trial>.mat``  28 labelled skin markers, 200 Hz, same clock
  ``gcRight/<trial>.mat``  the depositors' right-leg gait-cycle annotation, same clock
Each holds one MATLAB ``table`` object, which ``scipy.io.loadmat`` cannot decode; ``_read_table`` re-parses the
``__function_workspace__`` blob (columns in cell element 2, names in element 7). Nothing else is read.
``contacts_t`` holds the right heel strikes, read from ``gcRight`` where ``HeelStrike`` returns to zero.

Units: ``Header`` time in s (starts at about 12 s, identical in IMU and marker tables); accelerometer g -> m/s^2
(x 9.80665, specific force with gravity); gyroscope already rad/s; markers mm -> m.

Reference (``kind="skin"``): per segment a marker triad (origin at the first marker, x towards the second, z from
the third), with the four segment markers carried in ``Pose.markers``:
    thigh  R_Thigh_Upper -> R_Knee_Lat  -> R_Thigh_Front
    shank  R_Shank_Upper -> R_Ankle_Lat -> R_Shank_Front
    foot   R_Heel        -> R_Toe_Tip   -> R_Toe_Lat
Missing markers give NaN frames; no despiking or filtering. The world is +y up (turned to +z up by the loader).
The trunk has no markers, so its reference is None and no pair involves it.

Deposit defects left in the data: in AB06 ``treadmill_03_01`` the thigh and foot streams lead the markers by
2.4-2.5 s (``meta["defects"]``); marker spikes; markers gap-filled without a flag; inertial values resampled by an
undocumented amount; mounting undocumented; recordings do not start at rest; a residual per-segment lag of -1 to
+30 ms. A trial without a marker file is returned without references (``meta["defects"]["no_marker_file"]``).

Scope: only the ``treadmill`` mode by default (66 continuous recordings of 143-146 s); the other modes via
``Camargo(modes=("treadmill", "levelground", "ramp", "stair"))``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

from ..paths import dataset_dir
from ..schema import Adapter, Pair, Pose, Recording, Segment, SensorStream

G_MS2 = 9.80665
"""Standard gravity; converts the deposit's accelerometer from g to m/s².

Units: m/s².
Provenance: physical — the defined value of ISO 80000-3."""

MM_TO_M = 1e-3
"""Millimetres to metres; the deposit's marker table is in mm.

Units: m per mm.
Provenance: physical — an exact unit conversion."""

RATE_HZ = 200.0
"""Provenance: data-free — data-free."""

SENSOR_FAMILY = "Yost Labs (model number not stated anywhere in the deposit)"

SEGMENTS = ("trunk", "thigh", "shank", "foot")
"""The four instrumented segments, all on the right leg except the trunk (confirmed by the
angular-rate correlation of the foot unit against the right and left marker triads)."""

MARKERS = {
    "thigh": ("R_Thigh_Upper", "R_Thigh_Front", "R_Thigh_Rear", "R_Knee_Lat"),
    "shank": ("R_Shank_Upper", "R_Shank_Front", "R_Shank_Rear", "R_Ankle_Lat"),
    "foot": ("R_Heel", "R_Toe_Tip", "R_Toe_Med", "R_Toe_Lat"),
}
"""The four skin markers per referenced segment, carried verbatim on the ``Pose``."""

TRIADS = {
    "thigh": ("R_Thigh_Upper", "R_Knee_Lat", "R_Thigh_Front"),
    "shank": ("R_Shank_Upper", "R_Ankle_Lat", "R_Shank_Front"),
    "foot": ("R_Heel", "R_Toe_Tip", "R_Toe_Lat"),
}
"""The three markers the segment frame is built from, in the order expects."""

PAIRS = (
    ("thigh", "shank", "hinge", 1, "right knee"),
    ("shank", "foot", "2-dof", 2, "right ankle complex (talocrural plus subtalar)"),
)
"""The two connected pairs with a reference on both segments. The knee is one dominant degree of
freedom, the ankle two. ``trunk``–``thigh`` is not a joint (the pelvis lies between) and the trunk has no
reference, so it is not a pair."""

STREAM_OFFSET_TRIALS = {
    ("AB06", "treadmill_03_01"): {
        "stream_offset_seconds": 2.45,
        "segments": ["thigh", "foot"],
        "aligned_segments": ["shank"],
        "note": (
            "the thigh and foot inertial streams lead the markers by +2.4 to +2.5 s while the shank is "
            "aligned; measured by,. Exposed "
            "as deposited, not corrected."
        ),
    }
}
"""Provenance: measured on this dataset. Only 24
of the 66 treadmill trials were scanned, so this list is a lower bound on the prevalence.

Units: s.
Provenance: read-from-dataset — read from the deposit."""


def _read_table(source: Path | BinaryIO) -> dict[str, np.ndarray]:
    """Columns of a MATLAB v7 file holding one ``table`` object, as ``{name: 1-D float64 array}``.

    ``source`` is a path or an open binary stream (a member read out of a subject archive). The dataset probe's
    reader (``<dataset>/, with the SciPy 1.18 field rename handled.
    """
    top = sio.loadmat(source, struct_as_record=False)
    workspace = top["__function_workspace__"].tobytes()
    header = b" " * 116 + b"\x00" * 8 + b"\x00\x01" + b"IM"
    reader = MatFile5Reader(io.BytesIO(header + workspace[8:]), struct_as_record=False)
    reader.mat_stream.seek(0)
    reader.read_file_header()
    reader.initialize_read()
    var_header, _ = reader.read_var_header()
    opaque = reader.read_var_array(var_header, process=False)[0, 0].MCOS
    # SciPy < 1.18 names the three fields ("s0", "s1", "arr"); >= 1.18 uses ("_TypeSystem", "_Class",
    # "_ObjectMetadata"). The object metadata is the last field either way.
    cell = opaque[opaque.dtype.names[-1]][0]
    names = [str(x.flat[0]) for x in cell.flat[7].flat]
    columns = [np.asarray(v, dtype=np.float64).ravel() for v in cell.flat[2].flat]
    return dict(zip(names, columns, strict=True))


def _triad(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Right-handed frame from three marker tracks (N, 3): origin p1, x along p2 − p1, z from p3.

    Verbatim from the dataset probe's; the columns of the result are the segment axes
    expressed in the world frame, i.e. the matrix maps segment → world.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        e1 = p2 - p1
        e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        v = p3 - p1
        e3 = np.cross(e1, v)
        e3 = e3 / np.linalg.norm(e3, axis=1, keepdims=True)
        e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=2)


class Camargo(Adapter):
    """The treadmill recordings of Camargo et al. 2021, part 1 (subjects AB06–AB14)."""

    name = "camargo-treadmill"
    reference_kind = "skin"
    sensor_family = SENSOR_FAMILY
    licence = "CC BY 4.0 (as the Mendeley Data host page states it)"
    source = "doi:10.17632/fcgm3chfff.1 — see SOURCE.md in the dataset folder"

    def __init__(
        self, root: Path | None = None, archives: Path | None = None, modes: tuple[str, ...] = ("treadmill",)
    ) -> None:
        home = dataset_dir("camargo")
        self.root = Path(root) if root is not None else home / "_extracted"
        self.archives = Path(archives) if archives is not None else home
        self.modes = tuple(modes)
        self._members: dict[str, dict[str, str]] = {}

    def _subject_dirs(self) -> dict[str, Path]:
        """``{subject: <subject>/<date>}`` for the extracted tree; the date folder is unique per subject."""
        out: dict[str, Path] = {}
        if not self.root.is_dir():
            return out
        for subject_dir in sorted(p for p in self.root.glob("AB*") if p.is_dir()):
            for date_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name != "osimxml"):
                out[subject_dir.name] = date_dir
                break
        return out

    def _archive(self, subject: str) -> Path:
        return self.archives / f"{subject}.zip"

    def _archive_members(self, subject: str) -> dict[str, str]:
        """``{"<mode>/<kind>/<trial>": member name}`` for one subject archive, cached in memory.

        Reading a zip's central directory lists the members without decompressing any of them.
        """
        if subject not in self._members:
            index: dict[str, str] = {}
            path = self._archive(subject)
            if path.is_file():
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        parts = member.split("/")
                        if len(parts) == 5 and parts[-1].endswith(".mat"):
                            index[f"{parts[2]}/{parts[3]}/{parts[4][:-4]}"] = member
            self._members[subject] = index
        return self._members[subject]

    def _subjects(self) -> list[str]:
        extracted = set(self._subject_dirs())
        archived = {p.stem for p in self.archives.glob("AB*.zip")} if self.archives.is_dir() else set()
        return sorted(extracted | archived)

    def trials(self) -> list[tuple[str, str]]:
        """All ``(subject, trial)`` pairs, from file and archive member names only — no ``.mat`` is opened."""
        found: list[tuple[str, str]] = []
        subject_dirs = self._subject_dirs()
        for subject in self._subjects():
            date_dir = subject_dirs.get(subject)
            stems: set[str] = set()
            for mode in self.modes:
                imu_dir = None if date_dir is None else date_dir / mode / "imu"
                if imu_dir is not None and imu_dir.is_dir():
                    stems |= {p.stem for p in imu_dir.glob("*.mat")}
                else:
                    prefix = f"{mode}/imu/"
                    stems |= {k[len(prefix) :] for k in self._archive_members(subject) if k.startswith(prefix)}
            found.extend((subject, stem) for stem in sorted(stems))
        return found

    def _table(self, subject: str, trial: str, kind: str) -> dict[str, np.ndarray] | None:
        """One signal table of one trial, from the extracted tree if it is there, else from the subject archive.

        ``None`` when the deposit's file is reachable in neither place.
        """
        mode = trial.split("_")[0]
        date_dir = self._subject_dirs().get(subject)
        if date_dir is not None:
            path = date_dir / mode / kind / f"{trial}.mat"
            if path.is_file():
                return _read_table(path)
        member = self._archive_members(subject).get(f"{mode}/{kind}/{trial}")
        if member is None:
            return None
        with zipfile.ZipFile(self._archive(subject)) as archive:
            return _read_table(io.BytesIO(archive.read(member)))

    def load(self, subject: str, trial: str) -> Recording:
        imu = self._table(subject, trial, "imu")
        if imu is None:
            raise FileNotFoundError(
                f"{self.name}: {subject}/{trial}: no inertial table under {self.root} and none in "
                f"{self._archive(subject)}"
            )
        markers = self._table(subject, trial, "markers")
        gait_cycle = self._table(subject, trial, "gcRight")

        t_imu = imu["Header"]
        t_ref = None if markers is None else markers["Header"]

        segments: dict[str, Segment] = {}
        for name in SEGMENTS:
            stream = SensorStream(
                t=t_imu,
                gyr=np.stack([imu[f"{name}_Gyro_{a}"] for a in "XYZ"], axis=1),
                acc=np.stack([imu[f"{name}_Accel_{a}"] for a in "XYZ"], axis=1) * G_MS2,
                rate_hz=RATE_HZ,
                family=SENSOR_FAMILY,
                meta={
                    "deposited_units": {"acc": "g", "gyr": "rad/s", "t": "s"},
                    "note": "resampled onto the shared clock by an undocumented amount",
                },
            )
            reference = None
            if name in MARKERS and markers is not None:
                names = MARKERS[name]
                tracks = [np.stack([markers[f"{m}_{a}"] for a in "xyz"], axis=1) for m in names]
                stack = np.stack(tracks, axis=1) * MM_TO_M
                a, b, c = (stack[:, names.index(m)] for m in TRIADS[name])
                reference = Pose(
                    t=t_ref,
                    R=_triad(a, b, c),
                    p=a,
                    rate_hz=RATE_HZ,
                    kind="skin",
                    markers=stack,
                    marker_names=list(names),
                    meta={
                        "frame": f"marker triad {TRIADS[name]}, built as does",
                        "origin": TRIADS[name][0],
                        "deposited_units": "mm",
                        "gap_filled_by_depositors": True,
                        "despiked": False,
                    },
                )
            segments[name] = Segment(name=name, sensor=stream, reference=reference, anatomical=None, mount=None)

        pairs = [
            Pair(parent=parent, child=child, joint=joint, joint_dof=dof, meta={"plain_name": plain})
            for parent, child, joint, dof, plain in PAIRS
        ]

        defects: dict[str, object] = {
            "no_trunk_reference": "no torso markers; the OpenSim lumbar coordinates are identically zero",
            "no_marker_on_any_housing": "reference is a skin frame; the sensor-to-segment alignment is unobservable",
            "mounting_undocumented": True,
            "marker_spikes_not_removed": "pelvis cluster distorted >10 mm in 3.3 % of frames in the trial examined",
            "markers_gap_filled_without_a_flag": True,
            "inertial_streams_resampled": "no quantisation grid survives",
            "no_initial_rest_period": "still prefix median 0.01 s over the 66 treadmill trials; 30 of 66 have none",
            "residual_segment_lag_ms": "-1 to +30 ms against the markers, partly speed-dependent",
        }
        if (subject, trial) in STREAM_OFFSET_TRIALS:
            defects["stream_offset"] = STREAM_OFFSET_TRIALS[(subject, trial)]
        if markers is None:
            defects["no_marker_file"] = (
                "no marker table for this trial in the extracted tree or the subject archive, so every "
                "Segment.reference is None and only the sensor-side measures apply. On a complete checkout of "
                "part 1 this does not happen: the deposit ships one marker file per trial."
            )

        contacts_t = None
        if gait_cycle is not None:
            cycle = gait_cycle["HeelStrike"]
            starts = np.flatnonzero((cycle[1:] == 0) & (cycle[:-1] > 0)) + 1
            contacts_t = gait_cycle["Header"][starts]

        return Recording(
            dataset=self.name,
            subject=subject,
            trial=trial,
            segments=segments,
            pairs=pairs,
            same_segment_pairs=[],
            contacts_t=contacts_t,
            rest_annotation_s=None,
            trial_type=trial.split("_")[0],
            meta={
                "licence": self.licence,
                "source": self.source,
                "host": "Mendeley Data (part 1 of 3, subjects AB06-AB14)",
                "sensor_family": SENSOR_FAMILY,
                "reference": "Vicon optical motion capture, 200 Hz, skin markers on anatomical landmarks",
                "shared_clock": "the inertial and marker tables carry bit-identical Header columns",
                "side": "right leg",
                "contacts_source": (
                    "right-leg heel strikes, read as the zero-crossings of the depositors' gcRight HeelStrike "
                    "gait-cycle percentage; None when that table is not reachable"
                ),
                "no_deposited_joint_centre": "the jp table is joint powers, not joint positions",
                "defects": defects,
            },
        )


ADAPTER = Camargo()
