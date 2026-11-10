"""Loaders for the 14 datasets read through the dataset readers in `readers/` (one reader per dataset).

Each reader returns its own Recording (readers/characterisation/schema.py); this module translates it into this
package's Recording, so every check and correction works on them like on the hand-written loaders.

Translation, per trial:
    trial name      '<subject>/<reader trial>' (e.g. 'P01/overground/fast'); participant = subject.
    time            the reader gives every stream its own time stamps on one time base per trial. Here the
                    earliest stamp of all streams becomes 0 s and each stream keeps its own start (`start_s`), so
                    the pairing of IMU and reference is exactly the one the reader serves.
    IMU             gyroscope (rad/s) and accelerometer (m/s^2) as served; samples with NaN are invalid.
    reference       the reader's Pose.R, a rotation from the segment (sensor or cluster) frame to the world
                    frame, convention 'segment_to_world'; NaN frames are invalid.
    segments        only segments with both an IMU and a reference; the others are listed in
                    metadata['segments_left_out'].
    joints          the reader's pairs whose two segments are both kept.
    rate            constant-rate streams keep the reader's rate. A stream whose time stamps deviate from a
                    uniform grid by more than half a sample (only RepoIMU) is interpolated onto a uniform grid
                    over the same time span (linear for the IMU, spherical-linear for the reference; samples whose
                    stamp repeats an earlier one are left out), marked in metadata['resampled_from_irregular_clock'].
    empty end       trailing IMU samples with no reading in any gyroscope or accelerometer column are dropped:
                    every ULTRA-MoCap trial ends with one such row (time stamp, no readings), whose missing values
                    stop both filters; the count is in metadata['empty_imu_samples_dropped'].
    vertical        every reference world is taken as +z up (a constant rotation of the world cancels in the
                    relative orientation, so it is not handled as an error). Camargo and Central Florida walking
                    are +y up (raw markers: feet at y = 50-80 mm, thighs about 700 mm, Central Florida sternum
                    about 1,400 mm), so their reference orientations and marker positions are turned by 90 degrees
                    about x (WORLD_TO_Z_UP), which carries +y to +z.
    metadata        the reader's trial and stream metadata without large arrays, plus `timing` (how the reader
                    treats the time base; see TIMING) and `timing_as_deposited`.

Timing as served (TIMING): the readers serve every dataset's IMU-to-reference timing as the deposit pairs it, with
one exception. OpenAXES ships no common time base for the robot controller and the IMU host; its reader aligns the
two clocks by cross-correlation, which is the measurement the synchronisation-offset check makes, so its offset is
not a property of the deposit (timing_as_deposited False). For Trento upper limb (bonfiglio) and OpenSense the
readers can remove a measured lag; that removal is switched off.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .readers.characterisation.adapters import adapter as reader_adapter
from ..recording import ImuStream, Recording, ReferenceStream, Segment
from .base import DATA_ROOT, DatasetLoader

IRREGULAR_CLOCK_SAMPLES = 0.5  # time stamps further than this (in samples) from a uniform grid count as irregular
LARGE_ARRAY = 64  # metadata arrays with more elements than this are left out of the metadata

# Datasets whose reference world is not +z up, and the rotation that makes it so (applied on the left of the
# segment-to-world orientation and to the marker positions; module description, "vertical").
WORLD_TO_Z_UP = {name: Rotation.from_euler('x', 90, degrees=True) for name in ('camargo', 'wmcg')}

TIMING = {
    'bonfiglio': "optical time base as deposited (starts at 0 s, the sensors at their common SampleTimeFine "
                 "epoch); the reader's lag shift is switched off (APPLY_REFERENCE_LAG). The optical file of each "
                 "trial is paired by the reader's correlation-based assignment, not by index (trial pairing is "
                 "error 8).",
    'bonfiglio_deposited': "as bonfiglio, but each IMU trial is paired with the optical file of the same index, "
                           "as the deposit's own script assumes (error 8 as deposited).",
    'opensense': "reference rebuilt from the marker plates (readers/datasets/opensense); the raw IMU is placed "
                 "by the deposit's own orientation table; the reader's constant-lag removal is switched off "
                 "(REMOVE_CONSTANT_LAG).",
    'openaxes': "NOT as deposited: the deposit has no common time base for controller and IMU host; the "
                "reader aligns them by cross-correlation, so a measured offset is a residual of that "
                "alignment, not an error of the deposit.",
}


def _small(value):
    """`value` without numpy arrays of more than LARGE_ARRAY elements (recursively), for metadata."""
    if isinstance(value, dict):
        return {k: _small(v) for k, v in value.items() if not (isinstance(v, np.ndarray) and v.size > LARGE_ARRAY)}
    if isinstance(value, (list, tuple)) and len(value) > LARGE_ARRAY:
        return f'<{len(value)} values left out>'
    return value


def _irregularity_samples(t: np.ndarray, rate_hz: float) -> float:
    """Largest distance of the time stamps from the uniform grid t[0] + k / rate_hz, in samples."""
    return float(np.max(np.abs(t - (t[0] + np.arange(len(t)) / rate_hz))) * rate_hz) if len(t) > 1 else 0.0


def _increasing(t: np.ndarray) -> np.ndarray:
    """Mask of the samples whose time stamp is later than every earlier one (repeated stamps are left out of
    the interpolation; RepoIMU has a few)."""
    return np.concatenate([[True], t[1:] > np.maximum.accumulate(t)[:-1]])


def _uniform_grid(t: np.ndarray) -> tuple[np.ndarray, float]:
    rate = (len(t) - 1) / (t[-1] - t[0])
    return t[0] + np.arange(len(t)) / rate, float(rate)


def _empty_tail(gyr: np.ndarray, acc: np.ndarray) -> int:
    """Number of samples at the end without a reading in any gyroscope or accelerometer column."""
    empty = np.isnan(gyr).all(1) & np.isnan(acc).all(1)
    return len(empty) - (np.flatnonzero(~empty)[-1] + 1) if (~empty).any() else len(empty)


def _imu_stream(sensor, t0: float) -> tuple[ImuStream, float | None, int]:
    t, gyr, acc = np.asarray(sensor.t, float), np.asarray(sensor.gyr, float), np.asarray(sensor.acc, float)
    dropped = _empty_tail(gyr, acc)
    if dropped and dropped < len(t) - 1:  # module description, "empty end"
        t, gyr, acc = t[:-dropped], gyr[:-dropped], acc[:-dropped]
    else:
        dropped = 0
    rate, irregular = float(sensor.rate_hz), None
    if _irregularity_samples(t, rate) > IRREGULAR_CLOCK_SAMPLES:
        irregular = _irregularity_samples(t, rate)
        grid, rate = _uniform_grid(t)
        keep = _increasing(t)
        gyr = np.column_stack([np.interp(grid, t[keep], gyr[keep, i]) for i in range(3)])
        acc = np.column_stack([np.interp(grid, t[keep], acc[keep, i]) for i in range(3)])
    valid = np.isfinite(gyr).all(1) & np.isfinite(acc).all(1)
    return ImuStream(gyr, acc, rate, float(t[0] - t0), valid), irregular, dropped


def _turned(rotation: Rotation, positions: np.ndarray) -> np.ndarray:
    """Marker positions turned by a rotation; a frame with a missing coordinate stays missing."""
    out = rotation.apply(np.nan_to_num(positions, nan=0.0))
    out[np.isnan(positions).any(axis=1)] = np.nan
    return out


def _reference_stream(pose, t0: float, world_to_z_up: Rotation | None = None) -> tuple[ReferenceStream, float | None]:
    t, matrices = np.asarray(pose.t, float), np.asarray(pose.R, float)
    valid = np.isfinite(matrices).all(axis=(1, 2))
    matrices = np.where(valid[:, None, None], matrices, np.eye(3))
    orientation, rate, irregular = Rotation.from_matrix(matrices), float(pose.rate_hz), None
    if _irregularity_samples(t, rate) > IRREGULAR_CLOCK_SAMPLES:
        irregular = _irregularity_samples(t, rate)
        grid, rate = _uniform_grid(t)
        valid = valid & _increasing(t)
        if valid.sum() >= 2:
            inside = (grid >= t[valid][0]) & (grid <= t[valid][-1])
            rotvecs = np.zeros((len(grid), 3))
            rotvecs[inside] = Slerp(t[valid], orientation[np.flatnonzero(valid)])(grid[inside]).as_rotvec()
            nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
            valid = inside & valid[nearest]
            orientation = Rotation.from_rotvec(rotvecs)
        else:
            valid = np.zeros(len(grid), bool)
            orientation = Rotation.identity(len(grid))
    markers = None
    if pose.markers is not None and irregular is None:  # the raw markers the frame was built from, in mm (error 4)
        names = pose.marker_names or [f'marker{k + 1}' for k in range(pose.markers.shape[1])]
        markers = {n: 1000.0 * np.asarray(pose.markers[:, k], float) for k, n in enumerate(names)}
    if world_to_z_up is not None:
        orientation = world_to_z_up * orientation
        if markers is not None:
            markers = {n: _turned(world_to_z_up, m) for n, m in markers.items()}
    return ReferenceStream(orientation, rate, float(t[0] - t0), valid, 'segment_to_world', markers), irregular


def to_recording(reader_recording, name: str) -> Recording:
    """Translate one reader Recording into this package's Recording (see the module description)."""
    jr = reader_recording
    usable = {k: s for k, s in jr.segments.items()
              if s.sensor is not None and s.reference is not None and len(s.sensor.t) > 1 and len(s.reference.t) > 1}
    starts = [float(s.sensor.t[0]) for s in usable.values()] + [float(s.reference.t[0]) for s in usable.values()]
    t0 = min(starts) if starts else 0.0
    segments, segment_meta, irregular, empty_end = {}, {}, {}, {}
    for key, seg in usable.items():
        imu, irregular_imu, dropped = _imu_stream(seg.sensor, t0)
        if dropped:
            empty_end[key] = dropped
        reference, irregular_ref = _reference_stream(seg.reference, t0, WORLD_TO_Z_UP.get(name))
        segments[key] = Segment(key, imu, reference)
        segment_meta[key] = dict(reference_kind=seg.reference.kind, sensor_family=seg.sensor.family,
                                 native_rate_hz=seg.sensor.native_rate_hz, mount=seg.mount,
                                 imu=_small(seg.sensor.meta), reference=_small(seg.reference.meta))
        quaternion = seg.sensor.meta.get('vendor_quaternion_wxyz')
        if quaternion is not None and irregular_imu is None:  # Hip-ROM-Y: for restoring gravity (error 22)
            segment_meta[key]['vendor_quaternion_wxyz'] = np.asarray(quaternion, float)[:len(imu)]
            segment_meta[key]['acc_as_read'] = imu.acc.copy()
        for kind, value in (('imu', irregular_imu), ('reference', irregular_ref)):
            if value is not None:
                irregular[f'{key}.{kind}'] = value
    joints = [(p.parent, p.child) for p in jr.pairs if p.parent in segments and p.child in segments]
    left_out = {k: ('no IMU' if s.sensor is None or len(s.sensor.t) < 2 else 'no reference')
                for k, s in jr.segments.items() if k not in segments}
    metadata = dict(reader_dataset=jr.dataset, reader_subject=jr.subject, reader_trial=jr.trial,
                    trial_type=jr.trial_type, time_zero_s=t0, timing=TIMING.get(name, 'as deposited'),
                    timing_as_deposited=name != 'openaxes', segments_left_out=left_out,
                    resampled_from_irregular_clock=irregular, empty_imu_samples_dropped=empty_end, segments=segment_meta,
                    reader=_small(jr.meta),
                    joint_types={f'{p.parent}-{p.child}': (p.joint, p.joint_dof) for p in jr.pairs
                                 if p.parent in segments and p.child in segments})
    return Recording(name, f'{jr.subject}/{jr.trial}', segments, joints, metadata=metadata, participant=jr.subject)


class ReaderLoader(DatasetLoader):
    """A dataset read by one of the readers in `readers/`. Subclasses set `name` and `reader_name`."""

    reader_name: str = ''
    default_root = DATA_ROOT

    def __init__(self, root=None):
        if root is not None:
            raise ValueError(f'{self.name}: the reader finds its own data under {DATA_ROOT}; '
                             'set MOCAPREF_DATA to move it')
        super().__init__(None)

    @cached_property
    def reader(self):
        return reader_adapter(self.reader_name)

    @cached_property
    def _trial_keys(self) -> dict[str, tuple[str, str]]:
        return {f'{subject}/{trial}': (subject, trial) for subject, trial in self.reader.trials()}

    def trials(self) -> list[str]:
        return list(self._trial_keys)  # in the reader's order (by subject, then trial)

    def participant_of(self, trial: str) -> str:
        return self._trial_keys[trial][0]

    def session_of(self, trial: str) -> str:
        """The session of a trial: the trials recorded with the IMUs neither remounted nor switched off in between, as
        the dataset's documentation states it (both must be documented; plus Trento, whose sensor clocks run on across
        all trial files, and IS2B's blocks). Every other dataset: each trial its own
        session (the safe default of DatasetLoader).

        Geneva: one session per participant ("switched on about 10 minutes before the beginning of the recordings
        ... switched off after the end of recordings", one session of about one hour per participant; Grouvel 2023).
        Trento: one per participant ("instrumented with a set of 5 IMU ... Subsequently ... a sequence of 20
        exercises", Readme.docx; each DOT's SampleTimeFine clock runs on across the 20 files, measured);
        except subject 9's last trial, whose scapula file comes from another DOT device (2F605E402A6F), own session.
        IS2B: the authors' blocks of movement plane and range of motion (V/H/M x 60/110), each with its own
        gyroscope-bias recording in Data_Summary.xlsx; the functional trials (Fun) each their own."""
        subject, name = self._trial_keys[trial]
        if self.name in ('grouvel', 'grouvel_raw'):
            return subject
        if self.name in ('bonfiglio', 'bonfiglio_deposited'):
            return trial if trial == '9/t20-task-10' else subject
        if self.name == 'is2b':
            parts = name.split('_')  # plane_duration_excitation_range, e.g. H_120_f_110
            return trial if parts[1] == 'Fun' else f'{subject}/{parts[0]}_{parts[3]}'
        return trial

    def load(self, trial: str) -> Recording:
        rec = to_recording(self.reader.load(*self._trial_keys[trial]), self.name)
        rec.session = self.session_of(trial)
        return rec


class TrentoLoader(ReaderLoader):
    """Trento upper limb: besides the trials, every optical file of a participant can be read on its own, including
    the aborted and repeated takes no IMU trial is paired with (needed by the trial-pairing check, error 8)."""

    def optical_files(self, participant: str) -> list[str]:
        return [f'{participant}/optical-{j:02d}' for j in range(len(self.reader._optical_files(participant)))]

    def declared_optical(self, trial: str) -> str | None:
        """The optical file this reader pairs the trial with (as deposited: same index; otherwise the reader's
        correlation-based pairing)."""
        from .readers.characterisation.adapters import bonfiglio as trento
        subject, name = self._trial_keys[trial]
        k = int(name.split('-', 1)[0][1:]) - 1
        if self.reader.pairing == 'as_deposited':
            j = k if k < len(self.reader._optical_files(subject)) else None
        else:
            j = trento._session(str(self.reader.root), subject).pairing.get(k)
        return None if j is None else f'{subject}/optical-{j:02d}'

    def trial_imu_speeds(self, trial: str):
        """The gyroscope's rotation speed per segment straight from the deposited files, as the reader's pairing
        builds it, so a trial without any optical file still has one."""
        from ..errors.e08_trial_pairing import Speeds, _usable
        from .readers.characterisation.adapters import _bonfiglio_session as repair
        from .readers.characterisation.adapters import bonfiglio as trento
        subject, name = self._trial_keys[trial]
        streams = trento._read_trial_streams(self.reader._dot_files(subject)[int(name.split('-', 1)[0][1:]) - 1])
        rate, out = trento.RATE_HZ, {}
        for segment, stream in streams.items():
            grid = np.arange(stream['t'][0], stream['t'][-1], 1 / rate)
            if _usable(grid):
                g = np.stack([np.interp(grid, stream['t'], stream['gyr'][:, c]) for c in range(3)], axis=1)
                out[segment] = repair.zero_phase_lowpass(np.linalg.norm(g, axis=1), rate)
        return Speeds(rate, out)

    def optical_speeds(self, participant: str):
        """Rotation speed per segment of every optical file of the participant, built as the reader's pairing
        builds it (cluster Procrustes fit to the session template, one-step angular rate, gaps filled for
        the correlation only, 5 Hz low-pass); the magnitude does not depend on the world or cluster frame."""
        from ..errors.e08_trial_pairing import Speeds, _usable
        from .readers.characterisation.adapters import _bonfiglio_session as repair
        from .readers.characterisation.adapters import bonfiglio as trento
        session = trento._session(str(self.reader.root), participant)
        rate = trento.RATE_HZ
        out = {}
        for j, path in enumerate(self.reader._optical_files(participant)):
            markers = trento._read_optical_markers(path)
            speeds = {}
            for segment, (_, labels) in trento.SEGMENTS.items():
                if segment not in session.templates or not all(label in markers for label in labels):
                    continue
                cluster = np.stack([markers[label] for label in labels], axis=1)
                rotation, _ = repair.cluster_pose(cluster, session.templates[segment])
                _, w = repair.angular_rate(np.arange(len(rotation)) / rate, rotation)
                magnitude = repair.interpolate_gaps(np.linalg.norm(w, axis=1))
                if _usable(magnitude):
                    speeds[segment] = repair.zero_phase_lowpass(magnitude, rate)
            out[f'{participant}/optical-{j:02d}'] = Speeds(rate, speeds)
        return out


def _loader(name: str, reader_name: str, description: str) -> type[ReaderLoader]:
    base = TrentoLoader if reader_name.startswith('bonfiglio') else ReaderLoader
    return type(f'{name.title().replace("_", "")}Loader', (base,),
                dict(name=name, reader_name=reader_name, __doc__=description))


READER_LOADERS: list[type[ReaderLoader]] = [
    _loader('grouvel', 'grouvel-geneva', 'Geneva gait (Grouvel 2022): Physilog6S on trunk, pelvis and legs, '
                                         'printed housing clusters.'),
    _loader('fullbody17', 'fullbody-17imu', 'Hankyong full body: 17 Xsens IMUs with housing clusters.'),
    _loader('opensense', 'opensense-s2', 'OpenSense (Al Borno 2022): Xsens with machined marker plates, '
                                         'reference rebuilt from the markers.'),
    _loader('bonfiglio', 'bonfiglio-arm', 'Trento upper limb (Bonfiglio 2024): five Movella DOT on one arm, '
                                          'Optotrak clusters on the housings.'),
    _loader('bonfiglio_deposited', 'bonfiglio-arm-deposited',
            'Trento upper limb with the pairing as deposited: k-th IMU trial with k-th optical file.'),
    _loader('camargo', 'camargo-treadmill', 'Camargo lower limb (Georgia Tech): treadmill part, skin markers.'),
    _loader('wmcg', 'wmcg-ucf', 'Central Florida walking (WMCG): skin markers, raw trajectories.'),
    _loader('compliant_terrain', 'compliant-terrain', 'Delaware compliant terrain: Delsys IMUs, skin markers.'),
    _loader('grimmer_stairs', 'grimmer-stairs', 'Darmstadt stairs (Grimmer): skin markers.'),
    _loader('ultramocap', 'ultramocap-upper-limb', 'Central Florida upper limb (ULTRA-MoCap): OpenSim fit.'),
    _loader('hip_rom_y', 'hip-rom-y', 'Yanshan hip: vendor segment model.'),
    _loader('is2b', 'is2b', 'IS2B cadaver knee: bone pins with clusters.'),
    _loader('reemc', 'reemc', 'REEM-C humanoid robot.'),
    _loader('openaxes', 'openaxes', 'OpenAXES UR5 arm: encoder reference, clocks aligned by the reader.'),
    _loader('repoimu', 'repoimu-pendulum', 'RepoIMU triple pendulum (wand trials excluded).'),
]
