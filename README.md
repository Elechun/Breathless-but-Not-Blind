# Breathless-but-Not-Blind

**Interpreting Multi-Signal Patterns of Sleep Apnea with a Sensor-Robust Peripheral-Central Spiking Network**

Ki-Hun Jun, Sang-Ho Choi\* — Kwangwoon University · \*corresponding author (schoi@kw.ac.kr)
Poster, APCMBE / KOSOMBE 2026

[![poster](poster/poster_preview.png)](poster/APCMBE2026_Breathless_but_Not_Blind_poster.pdf)

Full-resolution poster: [`poster/APCMBE2026_Breathless_but_Not_Blind_poster.pdf`](poster/APCMBE2026_Breathless_but_Not_Blind_poster.pdf)

---

## TL;DR

A dendritic spiking neural network (DH-SNN) that reads seven PSG signals the way a clinician does: each signal gets its own **peripheral stream** with time constants matched to that signal, a **short-term-facilitation (STF) gate** decides which signals reach the **central** layer, and a **hold-update** readout carries evidence across minutes.

| Metric (CinC 2018 test, 1-min units, threshold 0.5) | Value |
|---|---|
| AUROC / AUPRC | **0.970 / 0.908** |
| Accuracy / F1 | 92.2 % / 0.820 |
| Sensitivity / Specificity | 85.0 % / 94.1 % |
| Parameters | 114 K |
| Activation sparsity | ≈ 88 % (firing rate 12.1 %) |

## Code

The main model of the poster (run `cinc_mm_stf_s0`) as a small PyTorch package, with the trained weights.

```
snn_apnea/model.py     network: per-signal encoder -> peripheral DH-LIF streams -> central STF layer -> hold-update readout
snn_apnea/data.py      CinC 2018 preprocessing (ECG beat series + SaO2 / airflow / belts at 4 Hz), labels, 7-min windows
snn_apnea/train.py     training and evaluation
splits/cinc2018_subject_split.json   subject-disjoint split, 694 / 99 / 201 subjects (AHI-stratified, seed 0)
checkpoints/cinc_mm_stf_s0.pt        trained poster model (114,135 parameters, 0.5 MB)
tests/check.py         quick CPU self-check
```

**Setup**

```bash
pip install -r requirements.txt
python tests/check.py                       # prints "ok"
```

**1. Preprocess** the [PhysioNet/CinC Challenge 2018](https://physionet.org/content/challenge-2018/1.0.0/) training set (994 records, one folder per record):

```bash
python -m snn_apnea.data --cinc-dir /path/to/challenge-2018/training --out data/cache --workers 16
```

**2. Evaluate the released checkpoint** on the 201 test subjects:

```bash
python -m snn_apnea.train --cache data/cache --eval checkpoints/cinc_mm_stf_s0.pt
python -m snn_apnea.train --cache data/cache --eval checkpoints/cinc_mm_stf_s0.pt --missing sao2   # 6.5: SaO2 sensor absent
```

**3. Train from scratch** (12 epochs, ~10 h and ~15 GB on one RTX A6000):

```bash
python -m snn_apnea.train --cache data/cache --out runs/main
```

GPU training is not bit-deterministic, so a new run lands close to, not exactly on, the released numbers.

**Inputs and window.** 7 signals at 4 Hz in the order `rri, ramp, edr, sao2, airflow, abd, chest`, each normalised per record. One sample is a 7-minute window: 4 past minutes, the scored minute, 2 later minutes for the hold-update step. A minute is labelled apnea when apnea/hypopnea events cover at least 5 s of it (RERA excluded).

**Released checkpoint on the test set** (93,266 minutes), produced by this code:

| | Acc | F1 | AUROC | AUPRC | Sen | Spe |
|---|---|---|---|---|---|---|
| all 7 signals, threshold 0.5 | 92.2 | 0.820 | 0.970 | 0.908 | 85.0 | 94.1 |
| SaO2 absent (`--missing sao2`), threshold 0.5 | 87.0 | 0.711 | 0.918 | 0.795 | 75.8 | 90.0 |

The poster's 6.5 sensitivity (75.4 %) silenced the SaO2 stream at the central input; `--missing` zeroes it at the encoder instead (same AUROC 0.918, sensitivity 75.8 %).

This package is a cleaned extract of the lab's research code. With the same seed it builds identical initial weights, and with the trained weights it gives the same outputs as the original model (fp32 bit-identical; bf16 test predictions within 6e-5).
Not included: the SaO2-frozen fine-tune of 6.5, the per-event-type reliance analysis of 6.2, and the figure scripts.

## 1–3. Introduction, dataset, setup

![intro](figures/01_introduction.png)

- **Data:** PhysioNet / CinC Challenge 2018 (MGH), 994 subjects, 459,368 one-minute units, apnea prevalence 20.9 %.
- **Signals (7):** Airflow, Chest, Abdomen, SaO2 (pulse oximetry), and three ECG-derived series (RRI, R-amplitude, QRS-area EDR).
- **Split:** subject-disjoint (one recording per subject), stratified by AHI: 694 train / 99 validation / 201 test subjects.

![dataset](figures/02_dataset.png)
![setup](figures/03_experimental_setup.png)

## 4. Architecture: peripheral → central

![architecture](figures/04_architecture.png)

| | |
|---|---|
| ![pipeline](figures/04-0_pipeline.png) | Overall pipeline |
| ![streams](figures/04-1_peripheral_streams.png) | **4.1 Peripheral streams.** Each signal has its own dendritic branches with their own time scales. |
| ![stf](figures/04-2_stf_gate_central.png) | **4.2 STF gate → central.** Signals that keep firing are boosted; unhelpful ones are suppressed before integration. |
| ![hold](figures/04-3_hold_update.png) | **4.3 Hold-update.** The readout keeps or revises its decision over consecutive minutes. |

## 5. Results I: what each mechanism does

![section 5](figures/05_results_mechanisms.png)

- **5.1 Matched time constants:** learning signal-matched τ instead of fixed τ improves per-signal AUROC: Airflow 0.699 → 0.780, Abdomen 0.776 → 0.876, Chest 0.762 → 0.839.
- **5.2 STF gate:** central neurons switch ON (132; 4.8 → 22.3 %) or OFF (93; 39.8 → 11.1 %) during apnea. STF gain: SaO2 0.63 → 0.77, Abdomen 0.55 → 0.70.
- **5.3 Hold-update:** Acc 91.8 → 92.2, F1 0.812 → 0.820, AUROC 0.966 → 0.970, AUPRC 0.900 → 0.908.

| ![5.1](figures/05-1_matched_time_constants.png) | ![5.2](figures/05-2_stf_gate_central_pattern.png) | ![5.3](figures/05-3_hold_update_effect.png) |
|---|---|---|

## 6. Results II: performance and robustness

![section 6](figures/06_results_performance.png)

**6.1 Baselines (CinC 2018):** best on every metric.

| Model | Input | Params | AUROC |
|---|---|---|---|
| **Ours (DH-SNN)** | 7 signals | 114 K | **0.970** |
| DRCNN | 12 PSG ch. incl. EEG | 934 K | 0.957 |
| SelANet (7 signals) | | | 0.947 |
| Transformer (Fayyaz et al., MLHC 2023) | multimodal | 149 K | 0.943 |
| SelANet (2 signals) | | | 0.927 |

![6.1](figures/06-1_baselines.png)

**6.2 Every event type is caught, and each relies on the signals clinicians use to score it (AASM).**
Recall: mixed 98.2 %, obstructive 90.9 %, central 84.8 %, hypopnea 81.4 %.
Signal reliance: obstructive uses Flow 38 / SaO2 19 / RRI 16 / Chest 14 / Abd 13 %; central and mixed use the effort belts (Chest 48 / Abd 41 %); hypopnea uses SaO2 (82 %).

![6.2](figures/06-2_event_types.png)

**6.3 One night (tr11-0064):** scored AHI 28.2 vs estimated 28.6.

![6.3](figures/06-3_one_night.png)

**6.4 Screening moderate-to-severe OSA (AHI ≥ 15):** 106 / 111 found (95.5 %), 82 / 90 correctly ruled out (91.1 %). The 5 misses scored 15.1–18.9; the 8 false alarms scored 11.3–14.8, so all errors sit near the cut-off.

![6.4](figures/06-4_ahi_screening.png)

**6.5 When SaO2 is missing at test time:** the STF model keeps 89 % of its sensitivity (85.0 → 75.4 %), compared with 35 % without STF and 49 % for a dense model. Fine-tuning with the SaO2 stream frozen raises AUROC from 0.918 to 0.935.

![6.5](figures/06-5_sao2_missing.png)

## 7–9. Conclusions and future work

![conclusions](figures/07-09_conclusions_future_work.png)

## Errata (corrections to the printed poster)

| Location | Printed | Correct |
|---|---|---|
| 2. Dataset | Normal 83.9 % / "Apena" 16.1 % | Normal 79.1 % / Apnea 20.9 % (1-min units; 16.1 % was the 30-s test split) |
| 4.3 | "next 4 units … g = 0.29" | next 2 units, g = 0.25 |
| 5.2 caption | "OFF on ON" | "OFF to ON" |
| 6.1 caption | "+0.1 %p and +0.45" | +1.0 %p (Acc) and +0.045 (F1) over the next best |
| Statistics | "subject-level paired bootstrap" | not performed for the 1-min results; differences are single-seed point estimates |

## Acknowledgement

This work was supported by the National Research Foundation of Korea (NRF), grant RS-2025-24533026.

---

© 2026 the authors. All rights reserved.
