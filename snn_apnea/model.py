"""Peripheral-central dendritic spiking network (DH-SNN) for per-minute sleep-apnea detection.

    x (B, 7, L) 4 Hz signals [rri, ramp, edr, sao2, airflow, abd, chest]
      -> per-signal conv encoder (grouped, no cross-signal mixing), 0.5 s frames (max + mean pooling)
      -> PERIPHERAL streams: one DH-LIF population per signal, 8 band-pass dendritic branches per neuron
      -> CENTRAL layer: STF gate on the peripheral spikes, one dendritic branch per signal, LIF soma
      -> leaky readout -> HOLD-UPDATE decision for the scored minute

Window: 7 minutes = 4 past minutes, the scored minute, 2 later minutes (hold-update).
Time step dt = 0.5 s, so a window is T = 840 steps.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGNALS = ("rri", "ramp", "edr", "sao2", "airflow", "abd", "chest")

# Main model (run cinc_mm_stf_s0). Band-pass branch k of a peripheral neuron passes periods between
# 2*pi*tau_fast[k] and 2*pi*tau_slow[k] (s); the 8 bands tile ~2 s .. ~400 s.
CONFIG = dict(
    in_channels=7, fs=4.0, frame_sec=0.5, context_min=7, center_index=4, hold_minutes=2,
    conv_dim=32, hidden_per_stream=36, branch=8, central_n=252, v_th=0.5, dropout=0.2, chan_drop=0.3,
    tau_fast=[[0.338, 0.397], [0.676, 0.794], [1.268, 1.489], [2.113, 2.481],
              [3.381, 3.969], [5.494, 6.45], [9.297, 10.914], [19.444, 22.825]],
    tau_slow=[[1.014, 1.191], [2.028, 2.381], [3.804, 4.466], [6.34, 7.443],
              [10.142, 11.907], [16.481, 19.349], [27.892, 32.743], [58.331, 68.475]],
    tau_m=[0.5, 4.0],                       # peripheral soma (s)
    central_tau_n=[5.0, 30.0], central_tau_m=[1.0, 4.0],
    stf_U=0.15, stf_tau=[1.0, 3.0], stf_kappa=4.0,
    readout_tau=[[5.0, 15.0], [20.0, 60.0]],
    seed=0,
)


# ----------------------------------------------------------------------------- spiking primitives

def _ncdf(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def spike(x: torch.Tensor) -> torch.Tensor:
    """Heaviside(x) forward; DH-SNN multi-Gaussian surrogate gradient backward (Zheng et al. 2024)."""
    s = (x >= 0).to(x.dtype)
    if not torch.is_grad_enabled():
        return s
    sig, h = 0.5, 0.15
    g = 0.5 * ((1 + h) * _ncdf(x / sig) - h * _ncdf((x - sig) / (6 * sig)) - h * _ncdf((x + sig) / (6 * sig)))
    return s + (g - g.detach())


def linear_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h[t] = a * h[t-1] + b[t] along dim 0 (h[-1] = 0, a constant in time), log-depth parallel scan."""
    s = 1
    while s < b.shape[0]:
        b = torch.cat([b[:s], b[s:] + a * b[:-s]], 0)
        a = a * a
        s *= 2
    return b


def lif_soma(l: torch.Tensor, alpha: torch.Tensor, v_th: float) -> torch.Tensor:
    """u[t] = alpha u[t-1] + (1 - alpha) l[t] - v_th s[t-1],  s[t] = H(u[t] - v_th).  l (T, ...) -> spikes."""
    mem = torch.zeros_like(l[0])
    s = torch.zeros_like(l[0])
    out = []
    for t in range(l.shape[0]):
        mem = alpha * mem + (1.0 - alpha) * l[t] - v_th * s
        s = spike(mem - v_th)
        out.append(s)
    return torch.stack(out, 0)


@torch.no_grad()
def init_tau_(p: torch.Tensor, ranges, dt: float, g: torch.Generator) -> None:
    """tau ~ LogUniform(lo, hi) seconds per column (last dim), stored as logit(exp(-dt / tau))."""
    flat = p.view(-1, p.shape[-1])
    for j, (lo, hi) in enumerate(ranges):
        u = torch.rand(flat.shape[0], generator=g, dtype=torch.float64)
        tau = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        beta = torch.exp(-dt / tau).clamp(1e-6, 1 - 1e-9)
        flat[:, j] = (torch.log(beta) - torch.log1p(-beta)).to(flat.dtype)


# ----------------------------------------------------------------------------- network blocks

class PeripheralStreams(nn.Module):
    """S independent DH-LIF populations (one per signal). Neuron i of stream s has K dendritic branches;
    branch k sees a fixed random 1/K of that stream's inputs and filters it with a band-pass
    d = LP_fast(I) - LP_slow(I) (two learnable poles); the soma sums the branches."""

    def __init__(self, S, n_in, H, K, tau_fast, tau_slow, tau_m, dt, v_th, seed):
        super().__init__()
        self.S, self.H, self.K, self.Cb, self.v_th = S, H, K, n_in // K, v_th
        perm, w, tm, tf, ts = [], [], [], [], []
        for s in range(S):
            g = torch.Generator().manual_seed(seed + 31 * s)
            perm.append(torch.randperm(n_in, generator=g))
            w.append(torch.empty(K, H, self.Cb).uniform_(-n_in ** -0.5, n_in ** -0.5, generator=g))
            g = torch.Generator().manual_seed(seed + 31 * s + 7919)
            a, b, c = torch.empty(H, 1), torch.empty(H, K), torch.empty(H, K)
            init_tau_(a, [tau_m], dt, g); init_tau_(b, tau_fast, dt, g); init_tau_(c, tau_slow, dt, g)
            tm.append(a[:, 0]); tf.append(b); ts.append(c)
        self.register_buffer("perm", torch.stack(perm))           # (S, n_in): branch k <- perm[s, k*Cb:(k+1)*Cb]
        self.weight = nn.Parameter(torch.stack(w))                 # (S, K, H, Cb)
        self.bn = nn.BatchNorm1d(S * K * H)
        self.tau_m = nn.Parameter(torch.stack(tm))                 # (S, H)      soma
        self.tau_n = nn.Parameter(torch.stack(tf))                 # (S, H, K)   fast pole
        self.tau_n2 = nn.Parameter(torch.stack(ts))                # (S, H, K)   slow pole

    def forward(self, f):                                          # (T, B, S*n_in) -> (T, B, S*H)
        T, B, _ = f.shape
        S, K, H = self.S, self.K, self.H
        x = f.view(T, B, S, -1)
        x = torch.gather(x, 3, self.perm.view(1, 1, S, -1).expand(T, B, S, -1)).view(T, B, S, K, self.Cb)
        I = torch.einsum("tbskc,skhc->tbskh", x, self.weight)
        I = self.bn(I.reshape(T * B, S * K * H)).view(T, B, S, K, H).transpose(3, 4)   # (T, B, S, H, K)
        bf, bs = torch.sigmoid(self.tau_n), torch.sigmoid(self.tau_n2)
        d = linear_scan(bf, (1.0 - bf) * I) - linear_scan(bs, (1.0 - bs) * I)
        return lif_soma(d.sum(-1).reshape(T, B, S * H), torch.sigmoid(self.tau_m).view(-1), self.v_th)


class CentralSTF(nn.Module):
    """Central integrator. STF gate: each peripheral spike is scaled by U + (1-U) tanh(kappa f), f = a
    leaky trace of that neuron's recent spikes (bursts pass, isolated spikes are attenuated).
    Each central neuron has one low-pass dendritic branch per signal stream, summed into a LIF soma."""

    def __init__(self, S, H, nc, dt, v_th, tau_n, tau_m, stf_U, stf_tau, stf_kappa, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n_in = S * H
        self.S, self.H, self.nc, self.v_th = S, H, nc, v_th
        self.stf_U = nn.Parameter(torch.full((n_in,), math.log(stf_U / (1 - stf_U))))
        self.stf_tau = nn.Parameter(torch.empty(n_in, 1)); init_tau_(self.stf_tau, [stf_tau], dt, g)
        self.stf_kappa = nn.Parameter(torch.full((n_in,), math.log(math.expm1(stf_kappa))))
        self.W = nn.Parameter(torch.randn(S, nc, H, generator=g) / math.sqrt(H))
        self.bn = nn.BatchNorm1d(S * nc)
        self.tau_n = nn.Parameter(torch.empty(nc, S)); init_tau_(self.tau_n, [tau_n] * S, dt, g)
        self.tau_m = nn.Parameter(torch.empty(nc, 1))
        init_tau_(self.tau_m, [tau_m], dt, torch.Generator().manual_seed(seed + 7919))

    def forward(self, s):                                          # (T, B, S*H) spikes -> (T, B, nc) spikes
        s = s.float()
        T, B, _ = s.shape
        rho = torch.sigmoid(self.stf_tau).view(1, 1, -1)
        f = linear_scan(rho, (1 - rho) * s)
        f = torch.cat([f.new_zeros(1, B, f.shape[2]), f[:-1]], 0)            # trace BEFORE the current spike
        U = torch.sigmoid(self.stf_U)
        x = (U + (1 - U) * torch.tanh(F.softplus(self.stf_kappa) * f)) * s
        I = torch.stack([x[:, :, k * self.H:(k + 1) * self.H] @ self.W[k].t() for k in range(self.S)], 2)
        I = self.bn(I.reshape(T * B, -1)).view(T, B, self.S, self.nc).transpose(2, 3)   # (T, B, nc, S)
        beta = torch.sigmoid(self.tau_n)
        l = linear_scan(beta, (1 - beta) * I).sum(-1)
        return lif_soma(l, torch.sigmoid(self.tau_m).view(-1), self.v_th)


class LeakyReadout(nn.Module):
    """y[t] = a y[t-1] + (1 - a) (W s[t] + b); channel 1 is the decision channel."""

    def __init__(self, n_in, n_out, dt, tau_ranges, seed):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        self.tau = nn.Parameter(torch.empty(1, n_out))
        init_tau_(self.tau, tau_ranges, dt, torch.Generator().manual_seed(seed))

    def forward(self, s):
        a = torch.sigmoid(self.tau).view(1, 1, -1)
        return linear_scan(a, (1.0 - a) * self.fc(s))


class SleepApneaSNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = dict(CONFIG, **(config or {}))
        C, e, dt = c["in_channels"], c["conv_dim"], c["frame_sec"]
        self.frame_len = int(round(c["fs"] * dt))
        self.spm = int(round(60 / dt))                              # SNN steps per minute
        self.center, self.n_min = c["center_index"], c["context_min"]
        self.hold_gain = nn.Parameter(torch.zeros(c["hold_minutes"]))
        self.hold_bias = nn.Parameter(torch.zeros(1))
        self.encoder = nn.Sequential(                               # groups=C: every signal has its own filters
            nn.Conv1d(C, C * (e // 2), 7, padding=3, groups=C, bias=False), nn.BatchNorm1d(C * (e // 2)), nn.ReLU(inplace=True),
            nn.Conv1d(C * (e // 2), C * e, 5, padding=2, groups=C, bias=False), nn.BatchNorm1d(C * e), nn.ReLU(inplace=True))
        H = c["hidden_per_stream"]
        self.peripheral = PeripheralStreams(C, 2 * e, H, c["branch"], c["tau_fast"], c["tau_slow"], c["tau_m"], dt,
                                            c["v_th"], c["seed"])
        self.central = CentralSTF(C, H, c["central_n"], dt, c["v_th"], c["central_tau_n"], c["central_tau_m"],
                                  c["stf_U"], c["stf_tau"], c["stf_kappa"], c["seed"] + 6060)
        self.readout = LeakyReadout(c["central_n"], 2, dt, c["readout_tau"], c["seed"] + 999)

    def forward(self, x, ctx_mask=None, missing=()):
        """x (B, C, L): context_min minutes at fs. ctx_mask (B, context_min) bool: False = padding beyond the
        record. missing: signal indices treated as disconnected (their encoder block is zeroed).
        Returns logits (B,) of the scored minute and logits_prov (B,), the causal provisional decision."""
        c = self.config
        B, C, L = x.shape
        T = L // self.frame_len
        h = self.encoder(x).reshape(B, C, -1, T, self.frame_len)   # (B, C, e, T, F)
        keep = torch.ones(B, C, device=x.device, dtype=h.dtype)
        if self.training and c["chan_drop"] > 0:                   # silence one whole signal for 30 % of samples
            sel = torch.rand(B, device=x.device) < c["chan_drop"]
            ch = torch.randint(0, C, (B,), device=x.device)
            keep[torch.arange(B, device=x.device)[sel], ch[sel]] = 0.0
        for k in missing:
            keep[:, k] = 0.0
        h = h * keep.view(B, C, 1, 1, 1)
        f = torch.stack([h.amax(-1), h.mean(-1)], 3).reshape(B, -1, T).permute(2, 0, 1)   # (T, B, C*2e)

        s = self.peripheral(f)
        if self.training and c["dropout"] > 0:                    # one mask per (sample, neuron), shared over time
            s = s * torch.bernoulli(s.new_full((1,) + s.shape[1:], 1 - c["dropout"])) / (1 - c["dropout"])
        y = self.readout(self.central(s))                          # (T, B, 2)

        # hold-update: provisional decision at the end of the scored minute, corrected by how the
        # decision moved over the next hold_minutes minutes (padded minutes keep the provisional value)
        end = lambda m: y[(m + 1) * self.spm - 1, :, 1]
        prov = end(self.center)
        later = torch.stack([end(self.center + j) for j in range(1, len(self.hold_gain) + 1)], 1)
        if ctx_mask is not None:
            later = torch.where(ctx_mask[:, self.center + 1:self.center + 1 + later.shape[1]].bool(), later, prov[:, None])
        logit = prov + ((later - prov[:, None]) * self.hold_gain).sum(1) + self.hold_bias
        return {"logits": logit, "logits_prov": prov}
