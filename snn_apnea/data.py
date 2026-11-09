"""PhysioNet/CinC Challenge 2018 -> 4 Hz model inputs and per-minute apnea labels.

Preprocessing (once per record, `python -m snn_apnea.data --cinc-dir <training dir> --out <cache dir>`):
  ECG (200 -> 100 Hz, 0.5-40 Hz band-pass, Elgendi R peaks) -> three beat series on a 4 Hz grid:
      rri  : cleaned RR interval            ramp : R-peak amplitude          edr : QRS-area EDR
  each robust-z normalised per record (median / IQR, clipped at +-6).
  SaO2 (pulse oximetry): values outside 50-100 % hold the last valid value, (x - 95) / 5.
  airflow, abdomen, chest: 0.05-1 Hz band-pass, 4 Hz, per-record robust z.
Labels: a minute is apnea if apnea/hypopnea events (central, obstructive, mixed, hypopnea; RERA
excluded) cover >= 5 s of it.
Writes <cache>/<record>.npz with x float32 (7, N4) in SIGNALS order, events float32 (n, 3) [onset s, offset s, type].
"""
from __future__ import annotations

import argparse
import json
import math
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.ndimage import median_filter

FS, SRC_FS, FEAT_FS = 100, 200, 4
FPM = 60 * FEAT_FS                                     # 4 Hz samples per minute
EVENT_TYPES = ("resp_centralapnea", "resp_obstructiveapnea", "resp_mixedapnea", "resp_hypopnea", "arousal_rera")
N_LABEL_TYPES = 4                                      # RERA is not part of the label
PSG = (("sao2", "SaO2"), ("airflow", "AIRFLOW"), ("abd", "ABD"), ("chest", "CHEST"))


# ----------------------------------------------------------------------------- ECG beat series

def bandpass(x, fs, lo, hi, order=4):
    sos = signal.butter(order, [lo / (0.5 * fs), min(hi / (0.5 * fs), 0.99)], btype="band", output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _window_max(x, centers, lo, hi):
    idx = np.clip(centers[:, None] + np.arange(lo, hi)[None, :], 0, len(x) - 1)
    seg = x[idx]
    a = seg.argmax(1)
    return idx[np.arange(len(centers)), a], seg[np.arange(len(centers)), a]


def elgendi_peaks(ecg, fs=FS, beta=0.08, polarity=0):
    """Elgendi et al. 2013 two-moving-average QRS detector (30 s local mean in the threshold)."""
    x = np.asarray(ecg, dtype=np.float64)
    y = bandpass(x, fs, 8.0, 20.0, order=3) ** 2

    def ma(v, w):
        w = max(int(round(w * fs)), 1)
        c = np.cumsum(np.concatenate([[0.0], v]))
        k = np.arange(len(v))
        lo, hi = np.clip(k - w // 2, 0, len(v)), np.clip(k + w - w // 2, 0, len(v))
        return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)

    blk = (ma(y, 0.12) > ma(y, 0.6) + beta * ma(y, 30.0)).astype(np.int8)
    d = np.diff(np.concatenate([[0], blk, [0]]))
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    wide = (off - on) >= int(0.08 * fs)
    on, off = on[wide], off[wide]
    if len(on) == 0:
        return np.zeros(0, np.int64)
    ecg_f = bandpass(x, fs, 0.5, 40.0)
    s = np.abs(ecg_f) if polarity == 0 else (ecg_f if polarity > 0 else -ecg_f)
    L = int((off - on).max())
    idx = np.minimum(on[:, None] + np.arange(L)[None, :], len(x) - 1)
    val = np.where(np.arange(L)[None, :] < (off - on)[:, None], s[idx], -np.inf)
    pk = idx[np.arange(len(on)), val.argmax(1)]
    keep = np.ones(len(pk), bool)                      # 250 ms refractory: keep the larger detection
    for i in np.flatnonzero(np.diff(pk) < int(0.25 * fs)):
        if s[pk[i]] >= s[pk[i + 1]]:
            keep[i + 1] = False
        else:
            keep[i] = False
    return pk[keep].astype(np.int64)


def estimate_polarity(ecg_f, beats, fs=FS):
    if len(beats) < 10:
        return 1
    lo, hi = int(-0.10 * fs), int(0.15 * fs)
    return 1 if np.median(_window_max(ecg_f, beats, lo, hi)[1]) >= np.median(_window_max(-ecg_f, beats, lo, hi)[1]) else -1


def refine_peaks(ecg_f, beats, polarity, fs=FS):
    if len(beats) == 0:
        return beats.astype(np.int64)
    pk = np.unique(_window_max(polarity * ecg_f, beats.astype(np.int64), int(-0.10 * fs), int(0.15 * fs))[0])
    keep = [0]                                         # merge peaks closer than 200 ms
    for i in range(1, len(pk)):
        if pk[i] - pk[keep[-1]] < int(0.2 * fs):
            if polarity * ecg_f[pk[i]] > polarity * ecg_f[pk[keep[-1]]]:
                keep[-1] = i
        else:
            keep.append(i)
    return pk[np.asarray(keep)].astype(np.int64)


def clean_rr(peaks, fs=FS):
    """RR intervals at the closing beat, with a validity mask (range, running-median ratio, ectopic pairs)."""
    t = peaks.astype(np.float64) / fs
    rr, t_rr = np.diff(t), t[1:]
    if len(rr) == 0:
        return t_rr, rr, np.zeros(0, bool)
    ok = (rr >= 0.3) & (rr <= 2.5)
    ref = median_filter(np.where(ok, rr, np.median(rr[ok]) if ok.any() else 1.0), size=11, mode="nearest")
    ok &= (rr >= 0.6 * ref) & (rr <= 1.5 * ref)
    if len(rr) > 2:
        a, b, r0 = rr[:-1], rr[1:], ref[:-1]
        ect = (a < 0.85 * r0) & (b > 1.15 * r0) & (np.abs(a + b - 2 * r0) < 0.25 * r0)
        spur = (a < 0.8 * r0) & (b < 0.8 * r0) & (np.abs(a + b - r0) < 0.2 * r0)
        irr = median_filter(np.abs(np.diff(rr)) / r0, size=21, mode="nearest")
        bad = (ect | spur) & (irr < 0.08)
        ok[:-1] &= ~bad
        ok[1:] &= ~bad
    return t_rr, rr, ok


def _interp(t, v, grid):
    if len(t) == 0:
        return np.zeros_like(grid)
    return np.full_like(grid, v[0]) if len(t) == 1 else np.interp(grid, t, v)


def robust_z(x, valid=None, clip=6.0):
    ref = x if valid is None or valid.sum() < 10 else x[valid]
    med = float(np.median(ref))
    iqr = float(np.subtract(*np.percentile(ref, [75, 25])))
    scale = iqr / 1.349 if iqr > 1e-9 else float(ref.std() + 1e-9)
    return np.clip((x - med) / scale, -clip, clip)


def ecg_series(x):
    """100 Hz ECG (mV) -> (3, N4) robust-z [rri, ramp, edr]."""
    n_grid = int(np.ceil(len(x) / (FS // FEAT_FS)))
    ecg_f = bandpass(x, FS, 0.5, 40.0)
    pol = estimate_polarity(ecg_f, elgendi_peaks(x, polarity=0)[:5000])
    peaks = refine_peaks(ecg_f, elgendi_peaks(x, polarity=pol), pol)
    grid = np.arange(n_grid, dtype=np.float64) / FEAT_FS
    t_rr, rr, ok = clean_rr(peaks)
    rri = _interp(t_rr[ok], rr[ok], grid)
    if ok.sum() >= 2:                                  # grid points supported by valid beats (gap <= 3 s)
        tv = t_rr[ok]
        k = np.searchsorted(tv, grid)
        gap = tv[np.clip(k, 0, len(tv) - 1)] - tv[np.clip(k - 1, 0, len(tv) - 1)]
        hr_ok = (grid >= tv[0]) & (grid <= tv[-1]) & (gap <= 3.0)
    else:
        hr_ok = np.zeros(n_grid, bool)
    s = pol * ecg_f
    pk = peaks[(peaks > int(0.15 * FS)) & (peaks < len(s) - int(0.1 * FS))]
    base = np.median(s[pk[:, None] + np.arange(int(-0.12 * FS), int(-0.06 * FS))[None, :]], axis=1) if len(pk) else np.zeros(0)
    amp = s[pk] - base
    area = (s[pk[:, None] + np.arange(int(-0.05 * FS), int(0.05 * FS) + 1)[None, :]] - base[:, None]).sum(1) / FS if len(pk) else np.zeros(0)
    if len(pk) > 20:                                   # beat-amplitude outliers -> running median
        for v in (amp, area):
            ref = median_filter(v, size=21, mode="nearest")
            mad = median_filter(np.abs(v - ref), size=61, mode="nearest") + 1e-6
            bad = np.abs(v - ref) > 5 * 1.4826 * mad
            v[bad] = ref[bad]
    raw = np.stack([rri, _interp(pk / FS, amp, grid), _interp(pk / FS, area, grid)]).astype(np.float32)
    return np.stack([robust_z(r, hr_ok) for r in raw])


# ----------------------------------------------------------------------------- one record

def _header(hea: Path):
    """{channel name: (row, gain, baseline)} of a WFDB header."""
    out = {}
    for i, line in enumerate(hea.read_text().splitlines()[1:]):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        g, base = p[2].split("/")[0], None
        if "(" in g:
            g, base = g.split("(")[0], float(g.split("(")[1].rstrip(")"))
        out[p[-1]] = (i, float(g), float(p[4]) if base is None else base)
    return out


def read_events(rec_dir: Path, rec: str) -> np.ndarray:
    import wfdb
    a = wfdb.rdann(str(rec_dir / rec), "arousal")
    fs = float(a.fs or SRC_FS)
    opened, out = {}, []
    for s, note in zip(a.sample, a.aux_note):
        m = re.match(r"^\((\w+)$", note or "")
        if m and m.group(1) in EVENT_TYPES:
            opened[m.group(1)] = int(s)
            continue
        m = re.match(r"^(\w+)\)$", note or "")
        if m and m.group(1) in opened:
            out.append((opened.pop(m.group(1)) / fs, int(s) / fs, EVENT_TYPES.index(m.group(1))))
    return np.asarray(sorted(out), dtype=np.float32).reshape(-1, 3)


def minute_labels(events: np.ndarray, n_min: int) -> np.ndarray:
    """1 if apnea/hypopnea events cover >= 5 s of the minute."""
    f = np.zeros(n_min, dtype=np.float32)
    for on, off, k in events:
        if int(k) >= N_LABEL_TYPES:
            continue
        for i in range(max(int(on // 60), 0), min(int(min(off, n_min * 60 - 1e-6) // 60), n_min - 1) + 1):
            f[i] += min(off, (i + 1) * 60.0) - max(on, i * 60.0)
    return (np.clip(f / 60.0, 0.0, 1.0) * 60.0 >= 5.0 - 1e-6).astype(np.int8)


def preprocess_record(rec_dir: Path, out_dir: Path) -> str:
    import scipy.io as sio
    from scipy.signal import resample_poly, butter, sosfiltfilt
    rec = rec_dir.name
    h = _header(rec_dir / f"{rec}.hea")
    sig = sio.loadmat(str(rec_dir / f"{rec}.mat"))["val"]
    phys = lambda name: (sig[h[name][0]].astype(np.float64) - h[name][2]) / h[name][1]
    ecg = resample_poly(phys("ECG"), FS, SRC_FS)
    feats = ecg_series(ecg)
    n4 = feats.shape[1]
    rows = []
    for key, name in PSG:
        x = phys(name)
        if key == "sao2":
            bad = (x < 50) | (x > 100)
            if bad.all():
                x[:] = 95.0
            elif bad.any():                            # hold the last valid value
                idx = np.where(~bad, np.arange(len(x)), 0)
                np.maximum.accumulate(idx, out=idx)
                idx[:np.argmax(~bad)] = np.argmax(~bad)
                x = x[idx]
            z = np.clip((resample_poly(x, FEAT_FS, SRC_FS).astype(np.float32) - 95.0) / 5.0, -6, 6)
        else:
            x = sosfiltfilt(butter(2, (0.05, 1.0), btype="band", fs=SRC_FS, output="sos"), x)
            z = robust_z(resample_poly(x, FEAT_FS, SRC_FS).astype(np.float32))
        rows.append(np.pad(z, (0, max(0, n4 - len(z))), mode="edge")[:n4])
    x = np.concatenate([feats[:, :n4], np.stack(rows)]).astype(np.float32)
    np.savez(out_dir / f"{rec}.npz", x=x, events=read_events(rec_dir, rec), n_min=len(ecg) // (60 * FS))
    return rec


# ----------------------------------------------------------------------------- windows

class Windows:
    """Every labelled minute of `records` as one sample: a context_min-minute window with the scored minute at
    index `center` (zero padded beyond the record). Data lives in one (rows, C, 240) tensor on `device`."""

    def __init__(self, records, cache: Path, context_min=7, center=4, stride=1, device="cpu"):
        self.M, self.h = context_min, center
        hL, hR = center, context_min - 1 - center
        rows, lab, ctx, start, rec_id, minute = [], [], [], [], [], []
        base = 0
        for i, r in enumerate(records):
            d = np.load(cache / f"{r}.npz")
            n = int(d["n_min"])
            x = d["x"][:, :n * FPM]
            x = np.pad(x, ((0, 0), (0, n * FPM - x.shape[1])))
            rows.append(np.concatenate([np.zeros((hL, x.shape[0], FPM), np.float32),
                                        x.reshape(x.shape[0], n, FPM).transpose(1, 0, 2),
                                        np.zeros((hR, x.shape[0], FPM), np.float32)]))
            y = minute_labels(d["events"], n)
            lab.append(np.r_[np.zeros(hL), y, np.zeros(hR)])
            ctx.append(np.r_[np.zeros(hL, bool), np.ones(n, bool), np.zeros(hR, bool)])
            m = np.arange(n)[::stride]
            start.append(base + m)
            rec_id.append(np.full(len(m), i))
            minute.append(m)
            base += hL + n + hR
        dev = torch.device(device)
        self.rows = torch.from_numpy(np.concatenate(rows)).to(dev)
        self.label = torch.from_numpy(np.concatenate(lab)).float().to(dev)
        self.ctx = torch.from_numpy(np.concatenate(ctx)).to(dev)
        self.start = torch.from_numpy(np.concatenate(start)).to(dev)
        self.record = torch.from_numpy(np.concatenate(rec_id))
        self.minute = torch.from_numpy(np.concatenate(minute))
        self.records = list(records)
        self.y = self.label[self.start + self.h].cpu()
        # record-balanced sampling weight: every record contributes equally
        n_r = np.bincount(self.record.numpy())
        self.weight = torch.from_numpy(1.0 / (len(records) * n_r[self.record.numpy()]))

    def __len__(self):
        return len(self.start)

    def batch(self, idx):
        idx = idx.to(self.start.device)
        rows = self.start[idx][:, None] + torch.arange(self.M, device=idx.device)
        x = self.rows[rows]                                        # (B, M, C, 240)
        return dict(x=x.permute(0, 2, 1, 3).reshape(len(idx), x.shape[2], -1),
                    y=self.label[rows[:, self.h]], ctx_mask=self.ctx[rows])

    def pos_weight(self) -> float:
        w = self.weight.double()
        pos = float((w * self.y.double()).sum())
        return (float(w.sum()) - pos) / pos


def load_split(path: Path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess CinC 2018 training records")
    ap.add_argument("--cinc-dir", type=Path, required=True, help="…/challenge-2018/training (one folder per record)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = sorted(p for p in a.cinc_dir.iterdir() if (p / f"{p.name}.mat").exists() and not (a.out / f"{p.name}.npz").exists())
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.starmap(preprocess_record, [(p, a.out) for p in recs])):
            if i % 50 == 0:
                print(f"{i + 1}/{len(recs)} {r}", flush=True)
