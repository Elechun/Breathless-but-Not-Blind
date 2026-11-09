"""Quick self-check (CPU, no data needed): python tests/check.py"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_apnea.data import minute_labels
from snn_apnea.model import SleepApneaSNN

# labels: >= 5 s of apnea/hypopnea in the minute; RERA (type 4) ignored; a 6 s event split 3 s + 3 s -> neither minute
ev = np.array([[10, 16, 1], [70, 73, 3], [130, 200, 4], [177, 183, 0]], np.float32)
assert minute_labels(ev, 4).tolist() == [1, 0, 0, 0]

m = SleepApneaSNN().eval()
assert sum(p.numel() for p in m.parameters()) == 114_135
ck = torch.load(Path(__file__).resolve().parents[1] / "checkpoints" / "cinc_mm_stf_s0.pt", map_location="cpu")
m.load_state_dict(ck["state_dict"])
x = torch.randn(2, 7, 7 * 240)
with torch.no_grad():
    a, b = m(x)["logits"], m(x)["logits"]
    c = m(x, missing=[3])["logits"]
assert a.shape == (2,) and torch.equal(a, b) and not torch.equal(a, c)
print("ok")
