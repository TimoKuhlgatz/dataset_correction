#!/usr/bin/env python
"""Rebuild the Geneva (grouvel) sensor streams on the optical clock from the raw 256 Hz session files.

Run once; the output folder is what the `grouvel_raw` loader reads (its default is
/home/data/imu-orientation/prepared/grouvel/raw-rebuild-v1):

    uv run python -m mocapref.datasets.readers.scripts.rebuild_geneva_from_raw --out <folder> [--subjects P01 ...]
        [--workers 10]

Why: the deposit's synchronised files carry sensor columns that are not the sensors' recordings of that trial on
the 100 Hz marker grid (mis-assembled streams, error 10). The raw `.BIN` files hold each sensor's whole session.
Per participant this script finds where every optically referenced trial lies in each sensor's session and
resamples the raw streams onto the trial's 100 Hz rows:

1. per sensor and trial, the reference's angular-rate magnitude (from the housing's marker cluster) is searched over
   the whole session by normalised cross-correlation;
2. the eight loggers keep one constant offset to each other over a session, so the trial's position is the median
   over the sensors; a sensor whose own search failed takes its position from that median and its own offset;
3. the position is refined per sensor to 1 ms, the raw streams are low-pass filtered (zero phase, 40 Hz) and
   interpolated at the rows' times, on the vendor's factory calibration from the file header.

Only time is fitted. Written per trial: one `.npz` with `acc/<segment>` (m/s^2) and `gyr/<segment>` (rad/s), and the
match table `match.csv`. With `--history-from <match.csv>` the script instead writes only the 60 s of stream that
precede each placed trial's first referenced row (lost pages are never bridged); `history.csv` records the length
per sensor-trial.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, firwin, resample_poly
from scipy.spatial.transform import Rotation

from mocapref.datasets.readers.characterisation.adapters._frames import DEG, STANDARD_GRAVITY
from mocapref.datasets.readers.characterisation.adapters.grouvel import NATIVE_RATE_HZ, RATE_HZ, GrouvelAdapter

LOWPASS_HZ = 40.0
"""Cut-off of the zero-phase low-pass before the 256 → 100 Hz interpolation. Units: Hz. Data-free: below the new
Nyquist frequency of 50 Hz with room for the filter's transition band."""
LOWPASS_TAPS = 101
RATE_WINDOW_S = 0.1
"""Window over which the reference's rotation and the gyroscope are averaged before their magnitudes are compared.
Units: s. Data-free: the marker clusters' frame-to-frame jitter is tens of °/s at 100 Hz, above the trunk's and
pelvis's own rate in walking; a tenth of a second is short against a stride."""
OWN_MATCH_CORR = 0.9
"""A sensor's own whole-session search counts as found above this correlation. Data-free."""
AGREE_S = 0.1
"""Sensors agree on a trial's position when within this of their median. Units: s. Data-free: a tenth of a stride."""
MIN_AGREEING = 3
FINE_HALF_S, FINE_STEP_S = 0.25, 0.001
MIN_REFERENCE_SAMPLES = 100
CALIBRATION_MIN_CORR = 0.99
"""A deposited trial counts as a clean shifted copy of the raw stream above this vector correlation. Data-free."""
CALIBRATION_SEARCH_S = 1.5
"""Half-width of the shift search between deposit and raw stream. Units: s. Data-free: beyond the deposit's
documented worst synchronisation error of 1.12 s."""
HISTORY_SECONDS = 60.0
"""Sensor history prepared before each trial's first referenced row (``--history-from``). Units: s."""
HISTORY_DEAD_RUN_S = 1.0
"""A run of identical consecutive samples at least this long inside the history is a dead logger stretch and the
history starts after it. Units: s."""


def rate_magnitude(R):
    """|mean angular rate over RATE_WINDOW_S| of a rotation-matrix series on the 100 Hz grid, centred; NaN where
    a frame is missing."""
    R = np.asarray(R, float)
    h = round(RATE_WINDOW_S * RATE_HZ / 2)
    ok = np.isfinite(R).all((1, 2))
    w = np.full(len(R), np.nan)
    good = ok[: -2 * h] & ok[2 * h :]
    rel = np.einsum("nji,njk->nik", R[: -2 * h][good], R[2 * h :][good])
    w[h:-h][good] = np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1) * RATE_HZ / (2 * h)
    return w


def windowed_magnitude(gyr):
    """The raw gyroscope's counterpart of rate_magnitude: |moving mean over RATE_WINDOW_S| at 256 Hz."""
    k = 2 * round(RATE_WINDOW_S * NATIVE_RATE_HZ / 2) + 1
    mean = np.stack([np.convolve(gyr[:, i], np.ones(k) / k, "same") for i in range(3)], 1)
    return np.linalg.norm(mean, axis=1)


def correlation_over(x, y):
    """Normalised cross-correlation of the template x (NaN = no weight) at every start in the long series y."""
    m = np.isfinite(x)
    n = m.sum()
    xc = np.where(m, x - x[m].mean(), 0.0)
    weight = m.astype(float)
    sy = fftconvolve(y, weight[::-1], "valid")
    syy = fftconvolve(y * y, weight[::-1], "valid")
    num = fftconvolve(y, xc[::-1], "valid")
    return num / (np.linalg.norm(xc) * np.sqrt(np.maximum(syy - sy**2 / n, 1e-12)))


def correlation_at(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return -1.0
    a, b = x[m] - x[m].mean(), y[m] - y[m].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else -1.0


def lowpass(x):
    """Zero-phase FIR low-pass of a (T, k) raw stream; NaN rows (lost logger pages) stay NaN and spread by the
    filter's half length, so a hole is never bridged."""
    taps = firwin(LOWPASS_TAPS, LOWPASS_HZ, fs=NATIVE_RATE_HZ)
    bad = ~np.isfinite(x).all(1)
    y = np.stack([np.convolve(np.where(bad, 0.0, x[:, i]), taps, "same") for i in range(x.shape[1])], 1)
    y[np.convolve(bad.astype(float), np.ones(LOWPASS_TAPS), "same") > 0] = np.nan
    return y


def sample_at(x, start_s, n):
    """The low-passed raw stream at the n rows of a trial that starts start_s into the sensor's session."""
    pos = (start_s + np.arange(n) / RATE_HZ) * NATIVE_RATE_HZ
    i0 = np.floor(pos).astype(int)
    f = (pos - i0)[:, None]
    ok = (i0 >= 0) & (i0 + 1 < len(x))
    out = np.full((n, x.shape[1]), np.nan)
    out[ok] = x[i0[ok]] * (1 - f[ok]) + x[i0[ok] + 1] * f[ok]
    return out


def history_start(acc, gyr):
    """First row of a history block that runs unbroken into the trial: after the last lost logger page (a NaN
    row, which also covers reaching before the session's first sample) and after the last dead stretch. A gap is
    never bridged, so the history simply starts later and is shorter than asked for."""
    bad = ~(np.isfinite(acc).all(1) & np.isfinite(gyr).all(1))
    start = int(np.flatnonzero(bad)[-1]) + 1 if bad.any() else 0
    n_run = round(HISTORY_DEAD_RUN_S * RATE_HZ)
    for x in (acc, gyr):
        same = (np.diff(x, axis=0) == 0).all(axis=1)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], same.astype(np.int8), [0]))))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s + 1 >= n_run:
                start = max(start, int(e) + 1)
    return start


def calibrated_streams(raw):
    """Every sensor's whole session, low-passed and on the vendor's factory calibration, with that calibration."""
    smooth, calibration = {}, {}
    for name, seg in raw.segments.items():
        calibration[name] = header_calibration(seg.sensor.meta["path"])
        smooth[name] = {kind: (lowpass(getattr(seg.sensor, kind)) - np.asarray(calibration[name][kind]["offset"]))
                        / np.asarray(calibration[name][kind]["scale"]) for kind in ("acc", "gyr")}
    return smooth, calibration


def history_participant(args):
    """The HISTORY_SECONDS of raw stream before each placed trial's first referenced row, on the trial's own
    100 Hz grid, from the placement this participant's rebuild recorded."""
    subject, out_dir, table, seconds = args
    smooth, _calibration = calibrated_streams(GrouvelAdapter().load_raw(subject))
    table = table[(table.subject == subject) & np.isfinite(table.start_s)]
    n_full = round(seconds * RATE_HZ)
    folder = Path(out_dir) / subject
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial, group in table.groupby("trial"):
        arrays = {}
        for _, row in group.iterrows():
            if row.segment not in smooth:
                continue
            block = {kind: sample_at(smooth[row.segment][kind], row.start_s - n_full / RATE_HZ, n_full)
                     for kind in ("acc", "gyr")}
            start = history_start(block["acc"], block["gyr"])
            for kind in ("acc", "gyr"):
                arrays[f"{kind}/{row.segment}"] = block[kind][start:]
            rows.append({"subject": subject, "trial": trial, "segment": row.segment, "start_s": row.start_s,
                         "history_samples": n_full - start, "history_s": (n_full - start) / RATE_HZ,
                         "gyr_median_deg_s": float(np.degrees(np.median(
                             np.linalg.norm(block["gyr"][start:], axis=1)))) if start < n_full else float("nan")})
        if arrays:
            np.savez(folder / f"{trial}.npz", **arrays)
    return pd.DataFrame(rows)


def header_calibration(path):
    """The vendor's factory calibration, read from the raw file's header page: after the accelerometer's tag
    (13 01 00 01) and the gyroscope's (14 01 00 03) stand three offsets and three scales as big-endian int32, in
    1e-4 g resp. 1e-4 °/s and in 1e-4; calibrated = (nominal − offset) / scale. The deposit's columns, where they
    are a clean copy of the raw stream, equal the raw stream under exactly this map (the gyroscope's scale of 0.87 is
    the sensor's 70 mdps per count against the nominal 2000 °/s / 32768)."""
    page = np.fromfile(path, dtype=np.uint8, count=512).tobytes()
    out = {}
    for kind, tag, unit in (("acc", bytes([0x13, 1, 0, 1]), 1e-4 * STANDARD_GRAVITY), ("gyr", bytes([0x14, 1, 0, 3]), 1e-4 * DEG)):
        at = page.index(tag) + 4
        values = np.frombuffer(page[at : at + 24], dtype=">i4").astype(float)
        out[kind] = {"offset": (values[:3] * unit).tolist(), "scale": (values[3:] * 1e-4).tolist()}
    return out


def deposit_check(table, smooth, deposited):
    """How the deposit's columns relate to the rebuilt streams: per unit and trial the best vector correlation of
    the gyroscopes over shifts of ±CALIBRATION_SEARCH_S, and, pooled per sensor over the trials that are clean
    shifted copies (at least CALIBRATION_MIN_CORR), the affine map from rebuilt to deposited (identity expected)."""
    pools = {}
    for i, row in table[np.isfinite(table.start_s)].iterrows():
        name = row.segment
        acc, gyr = deposited[(row.trial, name)]
        n = len(gyr)
        best = (-1.0, 0.0)
        for step, half, centre in ((0.01, CALIBRATION_SEARCH_S, 0.0), (0.001, 0.01, None)):
            centre = best[1] if centre is None else centre
            for shift in np.arange(centre - half, centre + half + step / 2, step):
                raw_gyr = sample_at(smooth[name]["gyr"], row.start_s + shift, n)
                ok = np.isfinite(raw_gyr).all(1)
                if ok.sum() < MIN_REFERENCE_SAMPLES:
                    continue
                c = np.corrcoef(raw_gyr[ok].ravel(), gyr[ok].ravel())[0, 1]
                if c > best[0]:
                    best = (float(c), float(shift))
        table.loc[i, ["deposit_corr", "deposit_shift_s"]] = best
        if best[0] < CALIBRATION_MIN_CORR:
            continue
        for kind, target in (("acc", acc), ("gyr", gyr)):
            raw = sample_at(smooth[name][kind], row.start_s + best[1], n)
            ok = np.isfinite(raw).all(1)
            pools.setdefault(name, {}).setdefault(kind, []).append((raw[ok], target[ok]))
    out = {}
    for name, kinds in pools.items():
        out[name] = {}
        for kind, pairs in kinds.items():
            x = np.concatenate([p[0] for p in pairs])
            y = np.concatenate([p[1] for p in pairs])
            solution = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)[0]
            out[name][kind] = {"matrix": solution[:3].T.tolist(), "offset": solution[3].tolist(),
                               "trials": len(pairs), "samples": len(x)}
    return out


def rebuild_participant(args):
    subject, out_dir = args
    ad = GrouvelAdapter()
    raw = ad.load_raw(subject)
    smooth, calibration = calibrated_streams(raw)
    coarse, magnitude = {}, {}
    for name in smooth:
        magnitude[name] = windowed_magnitude(smooth[name]["gyr"])
        coarse[name] = resample_poly(np.nan_to_num(magnitude[name]), 25, 64)  # 256 → 100 Hz, for the search only

    trials = [t for s, t in ad.trials() if s == subject]
    found, templates, lengths, deposited = [], {}, {}, {}
    for trial in trials:
        rec = ad.load(subject, trial)
        for name, seg in rec.segments.items():
            if seg.sensor is None or name not in raw.segments:
                continue
            lengths[(trial, name)] = len(seg.sensor.t)
            deposited[(trial, name)] = (np.asarray(seg.sensor.acc, float), np.asarray(seg.sensor.gyr, float))
            if seg.reference is None:
                continue
            w = rate_magnitude(seg.reference.R)
            if np.isfinite(w).sum() < MIN_REFERENCE_SAMPLES or np.nanstd(w) == 0.0:
                continue
            templates[(trial, name)] = w
            c = correlation_over(w, coarse[name])
            k = int(np.argmax(c))
            found.append({"trial": trial, "segment": name, "own_start_s": k / RATE_HZ, "own_corr": float(c[k])})
    found = pd.DataFrame(found)

    # each sensor's constant session offset to the anchor sensor, from the trials both found on their own
    good = found[found.own_corr >= OWN_MATCH_CORR]
    anchor = good.groupby("segment").own_corr.agg(["size", "median"]).sort_values(["size", "median"]).index[-1]
    wide = good.pivot(index="trial", columns="segment", values="own_start_s")
    offset = (wide.sub(wide[anchor], axis=0)).median()
    rows = []
    for trial, group in found.groupby("trial"):
        votes = group[group.own_corr >= OWN_MATCH_CORR]
        votes = votes.own_start_s.to_numpy() - offset.reindex(votes.segment).to_numpy()
        votes = votes[np.isfinite(votes)]
        centre = np.median(votes) if len(votes) else np.nan
        agreeing = int((np.abs(votes - centre) <= AGREE_S).sum()) if len(votes) else 0
        for _, row in group.iterrows():
            name, w = row.segment, templates[(trial, row.segment)]
            entry = {"subject": subject, "trial": trial, "segment": name, "own_start_s": row.own_start_s,
                     "own_corr": row.own_corr, "agreeing_sensors": agreeing, "start_s": np.nan, "corr": np.nan,
                     "source": "not found"}
            if agreeing >= MIN_AGREEING and np.isfinite(offset.get(name, np.nan)):
                expected = centre + offset[name]
                own = abs(row.own_start_s - expected) <= AGREE_S and row.own_corr >= OWN_MATCH_CORR
                shifts = np.arange(-FINE_HALF_S, FINE_HALF_S + FINE_STEP_S / 2, FINE_STEP_S)
                corr = [correlation_at(w, sample_at(magnitude[name][:, None], expected + s, len(w))[:, 0])
                        for s in shifts]
                best = int(np.argmax(corr))
                entry.update(start_s=expected + shifts[best], corr=corr[best], fine_shift_s=shifts[best],
                             source="own search" if own else "other sensors of the trial")
            rows.append(entry)
    table = pd.DataFrame(rows)

    # a weak sensor's refinement is not trusted on its own: it follows the trial's well-matched sensors
    for trial, group in table.groupby("trial"):
        strong = group[(group["corr"] >= OWN_MATCH_CORR)]
        if len(strong) < MIN_AGREEING:
            continue
        common = float(np.median(strong.fine_shift_s))
        weak = group.index[(group["corr"] < OWN_MATCH_CORR) & np.isfinite(group.start_s)]
        for i in weak:
            start = table.at[i, "start_s"] - table.at[i, "fine_shift_s"] + common
            w = templates[(trial, table.at[i, "segment"])]
            table.at[i, "start_s"], table.at[i, "fine_shift_s"] = start, common
            table.at[i, "corr"] = correlation_at(w, sample_at(magnitude[table.at[i, "segment"]][:, None], start, len(w))[:, 0])
            table.at[i, "source"] += ", refinement from the trial's strong sensors"

    check = deposit_check(table, smooth, deposited)

    folder = Path(out_dir) / subject
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calibration.json").write_text(json.dumps({"header": calibration, "deposit_against_rebuilt": check}, indent=1))
    table["nan_samples"] = 0
    for trial, group in table.groupby("trial"):
        arrays = {}
        for i, row in group.iterrows():
            if not np.isfinite(row.start_s):
                continue
            n = lengths[(trial, row.segment)]
            for kind in ("acc", "gyr"):
                arrays[f"{kind}/{row.segment}"] = sample_at(smooth[row.segment][kind], row.start_s, n)
            table.at[i, "nan_samples"] = int((~np.isfinite(arrays[f"gyr/{row.segment}"]).all(1)).sum())
        if arrays:
            np.savez(folder / f"{trial}.npz", **arrays)
    table.to_csv(folder / "match.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--history-from", type=Path, default=None,
                        help="match.csv of a rebuild: write only the sensor history before each placed "
                             "trial, not the trials themselves")
    parser.add_argument("--history-seconds", type=float, default=HISTORY_SECONDS)
    args = parser.parse_args()
    out = args.out.expanduser()
    subjects = args.subjects or GrouvelAdapter().raw_subjects()
    if args.history_from is not None:
        out.mkdir(parents=True, exist_ok=True)
        placed = pd.read_csv(args.history_from.expanduser())
        with Pool(min(args.workers, len(subjects))) as pool:
            tables = pool.map(history_participant,
                              [(s, out, placed, args.history_seconds) for s in subjects])
        table = pd.concat(tables, ignore_index=True)
        table.to_csv(out / "history.csv", index=False)
        full = round(args.history_seconds * RATE_HZ)
        short = table[table.history_samples < full]
        print(f"{len(table)} sensor-trials of {table.trial.nunique()} trials; history {table.history_s.min():.2f}–"
              f"{table.history_s.max():.2f} s, median {table.history_s.median():.2f} s; "
              f"{len(short)} sensor-trials shorter than {args.history_seconds:.0f} s")
        if len(short):
            print(short[["subject", "trial", "segment", "history_s"]].to_string())
        return
    with Pool(min(args.workers, len(subjects))) as pool:
        tables = pool.map(rebuild_participant, [(s, out) for s in subjects])
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "match.csv", index=False)
    table["type"] = table.trial.str.replace(r"_\d+$", "", regex=True)
    print(table.groupby("type").agg(units=("corr", "size"), placed=("start_s", lambda v: int(np.isfinite(v).sum())),
                                    corr_median=("corr", "median"), corr_p05=("corr", lambda v: v.quantile(0.05)),
                                    nan_units=("nan_samples", lambda v: int((v > 0).sum()))).round(3).to_string())
    print(table.source.value_counts().to_string())


if __name__ == "__main__":
    main()
