# Multimodal Embedding Audit (autoresearch_agent-c-20260407_215927)

## Data and setup

- Source file: `/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/adamgab/code_projects/multimodal_research_agent_c/embeddings/autoresearch_agent-c-20260407_215927/embeddings.h5`
- Shape: `embeddings = (240 visits, 3 modalities, 256 dims)`
- Modalities detected: `cgm`, `retina`, `metabolites`
- Longitudinal coverage: `120` baseline -> `02_00_visit` pairs (subject-level)
- Eval protocol reproduced: 5-fold subject-wise Ridge probes, compared to `benchmark_results.csv`

## Core benchmark result (what "how good" means here)

- Benchmark rerun now uses `age/gender/bmi` as predictors in tabular baselines (per your request).
- `EVAL_SCORE = -0.0774` for embedding-only probes under this stronger baseline.
- `EVAL_SCORE = -0.0764` after adding `age/gender/bmi` to embedding probes.
- Task means (embedding+AGB): `Task1 mean r = 0.2125`, `Task2 mean r = 0.2502`.
- Interpretation: demographics give only a small gain, while the benchmark baseline gets stronger, so gap widens.

## AGB effect size (your suspicion test)

- Adding `age/gender/bmi` to embedding probes improves:
  - `EVAL_SCORE` by `+0.0010`
  - Task1 mean r by `+0.0020`
  - Task2 mean r by `+0.0000` (negligible)
- Conclusion: AGB matters, but only marginally for this embedding set.

## KPI-level performance vs baseline

### Strong positives

- **Task1 cross-modal retina KPI transfer is strong**:
  - `sitting_blood_pressure_systolic`: `+0.271` delta r
  - `intima_media_th_mm_1_intima_media_thickness`: `+0.286` delta r
- **Task2 longitudinal intima prediction is strong**:
  - `intima_media_th_mm_1_intima_media_thickness`: `+0.233` delta r

### Near parity / small gain

- `bt__glucose`: small positive deltas in both tasks (`+0.036`, `+0.033`)
- Bilirubin task1: slight gain (`+0.054`)

### Main weaknesses

- `bt__hba1c` is substantially below baseline in both tasks (`-0.380`, `-0.147`)
- `urate` longitudinal is strongly below baseline (`-0.471`)
- Bilirubin longitudinal is below baseline (`-0.283`)

## What embeddings seem to encode

- They carry **cross-modal shared state** very strongly:
  - Same-visit top-1 retrieval is far above chance (`0.47`-`0.83` vs chance `0.0042`)
  - Highest pairwise coupling is `retina <-> metabolites` (`0.817` and `0.829`)
- They also preserve **subject identity over time**:
  - Baseline -> follow-up top-1 ID is `0.15` (CGM), `0.258` (retina), `0.267` (metabolites), all >> chance `0.0083`

## Novel finding

The strongest novel signal is a **retina-metabolites latent coupling**:

- Retina and metabolites embeddings can retrieve each other at the same visit with ~`0.82` top-1 accuracy across `240` visits.
- This is much higher than chance and higher than other modality pairs, suggesting the backbone learns a shared biological axis most visible in these two modalities.
- Practical implication: a targeted retrieval or distillation objective between retina and metabolites could improve transfer for cardio-metabolic KPIs.

## Selected "cool" probes beyond KPI set

Using the same subject-wise CV setup with concatenated embeddings:

- **Strongly encoded**:
  - `body_comp_total_lean_mass`: `r=0.851` (embedding+AGB)
  - `total_scan_vat_area`: `r=0.800`
  - `iglu_mage`: `r=0.741`
  - `iglu_cv`: `r=0.723`
  - `iglu_hbgi`: `r=0.708`
- **Moderate signal**:
  - `qrs_ms`: `r=0.332`
  - `hr_bpm`: `r=0.265`
- **Weak/negative**:
  - `total_scan_sat_mass`: `r=-0.146`

This indicates the representation is already strong for glycemic variability and some body-composition axes, even if KPI benchmark delta remains negative overall.

## Caveats

- Small sample size (`120` subjects, `240` visits) causes fold volatility.
- Training metadata indicates very short pretraining (`epochs=1`, `max_steps=2000`), likely undertrained.
- `concat_all` longitudinal ID retrieval is worse than best single modalities, suggesting simple concatenation may add noise without alignment weighting.

## Files produced

- Main script: `research/multimodal_embedding_audit_20260407/analysis.py`
- PDF report: `research/multimodal_embedding_audit_20260407/report.pdf`
- Tables: `metrics.json`, `eval_vs_baseline_summary.csv`, `subject_fingerprint.csv`, `cross_modal_retrieval.csv`, `longitudinal_delta_signal.csv`
- Figures: `research/multimodal_embedding_audit_20260407/figures/*.png`
