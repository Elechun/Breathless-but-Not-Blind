"""Train or evaluate the peripheral-central DH-SNN on CinC 2018 (subject-disjoint split, 1-minute units).

train:     python -m snn_apnea.train --cache data/cache --out runs/main
evaluate:  python -m snn_apnea.train --cache data/cache --eval checkpoints/cinc_mm_stf_s0.pt [--missing sao2]

Training (the poster model): 12 epochs, batch 64, every 4th minute of each train record, record-balanced
sampling; AdamW (time constants, BN, STF and hold parameters lr 5e-3 / wd 0, the rest lr 1e-3 / wd 0.05),
1-epoch warm-up + cosine; BCE(pos_weight) on the hold-update logit + 0.5 x BCE on the provisional logit;
bf16 autocast. The epoch with the best validation accuracy at 0.5 is kept.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import Windows, load_split
from .model import SIGNALS, SleepApneaSNN

ROOT = Path(__file__).resolve().parents[1]


def metrics(y, p, thr=0.5):
    y, pred = np.asarray(y).astype(int), np.asarray(p) >= thr
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    return dict(acc=(tp + tn) / len(y), f1=2 * tp / max(2 * tp + fp + fn, 1), sens=tp / max(tp + fn, 1),
                spec=tn / max(tn + fp, 1), auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), n=len(y))


def best_f1_threshold(y, p):
    o = np.argsort(-p)
    ps, ys = p[o], y[o]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    f1 = 2 * tp / np.maximum(tp + fp + ys.sum(), 1)
    return float(ps[int(np.argmax(np.where(np.r_[ps[1:] != ps[:-1], True], f1, -1)))])


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict(model, ds, device, bs=256, missing=()):
    model.eval()
    ps = []
    for i in range(0, len(ds), bs):
        b = ds.batch(torch.arange(i, min(i + bs, len(ds))))
        with amp(device):
            ps.append(torch.sigmoid(model(b["x"], b["ctx_mask"], missing)["logits"].float()).cpu())
    return torch.cat(ps).numpy(), ds.y.numpy().astype(int)


def make_optimizer(model, lr, tau_lr, wd):
    special = lambda n: (n.split(".")[-1] in ("tau", "tau_n", "tau_n2", "tau_m", "stf_U", "stf_tau", "stf_kappa", "hold_gain", "hold_bias")
                         or ".bn." in n or n.startswith("encoder.1.") or n.startswith("encoder.4."))
    groups = [{"params": [p for n, p in model.named_parameters() if not special(n)], "lr": lr, "weight_decay": wd, "base_lr": lr},
              {"params": [p for n, p in model.named_parameters() if special(n)], "lr": tau_lr, "weight_decay": 0.0, "base_lr": tau_lr}]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def train(a, device):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    split = load_split(a.split)
    tr = Windows(split["train"], a.cache, stride=4, device=device)
    va = Windows(split["val"], a.cache, device=device)
    print(f"train {len(tr)} windows / val {len(va)}", flush=True)
    model = SleepApneaSNN({"seed": a.seed}).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = make_optimizer(model, a.lr, a.tau_lr, a.wd)
    pos_weight = torch.tensor(tr.pos_weight(), device=device)
    bce = lambda z, y: F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
    steps = len(tr) // a.batch_size
    total, warm = steps * a.epochs, steps
    a.out.mkdir(parents=True, exist_ok=True)
    best, step = (-1.0, -1.0), 0
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(a.seed * 100003 + ep)
        order = torch.multinomial(tr.weight, len(tr), replacement=True, generator=g)
        tot = 0.0
        for i in range(steps):
            f = (step + 1) / warm if step < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
            for grp in opt.param_groups:
                grp["lr"] = grp["base_lr"] * f
            b = tr.batch(order[i * a.batch_size:(i + 1) * a.batch_size])
            with amp(device):
                o = model(b["x"], b["ctx_mask"])
            loss = bce(o["logits"].float(), b["y"]) + 0.5 * bce(o["logits_prov"].float(), b["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot += loss.item()
        p, y = predict(model, va, device)
        m = metrics(y, p)
        better = (m["acc"], m["auroc"]) > best
        print(f"[ep {ep:02d}] loss {tot / steps:.4f}  val acc {m['acc']:.4f} auroc {m['auroc']:.4f}  ({time.time() - t0:.0f}s){'  *' if better else ''}", flush=True)
        if better:
            best = (m["acc"], m["auroc"])
            torch.save({"config": model.config, "state_dict": model.state_dict(), "epoch": ep}, a.out / "best.pt")
    ck = torch.load(a.out / "best.pt", map_location="cpu")
    model.load_state_dict(ck["state_dict"])
    p, y = predict(model, va, device)
    ck["threshold"] = best_f1_threshold(y, p)                     # max-F1 threshold on validation (secondary)
    torch.save(ck, a.out / "best.pt")
    return a.out / "best.pt"


def evaluate(ckpt, a, device):
    ck = torch.load(ckpt, map_location="cpu")
    model = SleepApneaSNN(ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    te = Windows(load_split(a.split)["test"], a.cache, device=device)
    missing = [SIGNALS.index(s) for s in a.missing]
    p, y = predict(model, te, device, missing=missing)
    res = {"threshold_0.5": metrics(y, p, 0.5), f"threshold_val_maxF1({ck.get('threshold', 0.5):.3f})": metrics(y, p, ck.get("threshold", 0.5)),
           "missing": a.missing}
    print(json.dumps(res, indent=1))
    if a.save_pred:
        np.savez_compressed(a.save_pred, p=p, y=y, record=np.asarray(te.records)[te.record.numpy()], minute=te.minute.numpy())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="output dir of python -m snn_apnea.data")
    ap.add_argument("--split", type=Path, default=ROOT / "splits" / "cinc2018_subject_split.json")
    ap.add_argument("--out", type=Path, default=Path("runs/main"))
    ap.add_argument("--eval", type=Path, help="evaluate this checkpoint on the test records instead of training")
    ap.add_argument("--missing", nargs="*", default=[], choices=SIGNALS, help="signals treated as disconnected at test time")
    ap.add_argument("--save-pred", type=Path)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau-lr", type=float, default=5e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    evaluate(a.eval if a.eval else train(a, device), a, device)


if __name__ == "__main__":
    main()
