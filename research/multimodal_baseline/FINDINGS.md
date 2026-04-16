# Multimodal Baseline Benchmark — FINDINGS

## Overview

Benchmark dataset for evaluating multimodal health representation learning.
120 subjects with 4 modalities across exactly 2 visits, 6 KPIs — **100% KPI coverage on every row**.
Evaluation: **5-fold subject-level cross-validation** (random, seed=42), 96 train / 24 test subjects per fold.

---

## Dataset Design

### Modalities

| Modality | Raw Signal | Tabular Features Used | N features |
|---|---|---|---|
| **CGM** | FreeStyle Libre txt files | glycemic_status (46 features) | 46 |
| **DEXA** | H5 scan images (bone/tissue) | body_composition + bone_density | 294 |
| **Retina** | H5 fundus images (1024×1024 OD+OS) | cardiovascular_system (proxy) | 128 |
| **Metabolites** | — | metabolites_annotated (mass-spec) | 172 |

> **Retina note:** No pre-computed AVR or RNFL features exist in the dataset.
> Cardiovascular features (systolic BP, carotid IMT) are used as retinal-vascular proxies —
> they capture the same microvascular/atherosclerotic axis visible in fundus images.

### KPIs (6 total, 2 per modality for CGM, retina proxy, and metabolites)

| Modality | KPI | Clinical Meaning |
|---|---|---|
| CGM | `bt__hba1c` | HbA1c from blood test — long-term glycemic control |
| CGM | `bt__glucose` | Fasting glucose from blood test |
| Retina (proxy) | `sitting_blood_pressure_systolic` | Systolic BP — vascular health (AVR proxy) |
| Retina (proxy) | `intima_media_th_mm_1_intima_media_thickness` | Carotid IMT — atherosclerosis (vascular aging) |
| Metabolites | `urate` | Uric acid — metabolic syndrome, gout, CVD risk |
| Metabolites | `Bilirubin` | Heme catabolism / liver oxidative stress — antioxidant capacity |

> **DEXA note:** DEXA features (body_composition + bone_density, 294 features) are used as **inputs** to predict other KPIs, but DEXA KPIs (`total_scan_vat_mass`, `body_total_bmd`) are not prediction targets.

> **CGM KPI note:** `iglu_gmi` and `iglu_in_range_70_180` require a worn CGM device —
> only a handful of subjects have these at 2+ visits. `bt__hba1c` and `bt__glucose` come from
> blood tests taken at every visit and have complete coverage.

### 120 Subject Cohort

**Eligibility:** ALL 6 KPIs non-null at baseline AND at 02_00_visit (hard requirement).
Also: retina any visit, multi-visit DEXA, multi-visit metabolites, CGM any visit.

Stratified by **age × gender** (3 age bins × 2 genders = 6 cells, 20 each):

| Age bin | F | M |
|---|---|---|
| 40–50 | 20 | 20 |
| 50–60 | 20 | 20 |
| 60–70 | 20 | 20 |

- Mean age: 54.3 ± 8.2 years | Mean BMI: 25.9 ± 3.5 kg/m²
- Gender: 60F / 60M
- **Train: 96 subjects (80%) | Test: 24 subjects (20%)** — subject-level split
- **240 total rows: 120 baseline + 120 at 02_00_visit** — every row has all 6 KPIs

### KPI Coverage

**100% non-null across all 240 rows for all 6 KPIs.** This is enforced by subject eligibility
and row-level filtering (rows with any missing KPI are excluded).

---

## Methods

### Subject Selection

Subjects were drawn from the HPP 10K cohort. Eligibility required: (1) all 6 KPIs non-null at both baseline and 02_00_visit; (2) retina scan at any visit; (3) DEXA scan at 2+ visits; (4) metabolomics at 2+ visits; (5) CGM data at any visit; (6) age 40–70 years with non-missing age, gender, and BMI. The eligible pool comprised 296 subjects across all age×gender strata.

A stratified sample of 20 subjects per age-bin × gender cell (3 × 2 = 6 cells) was drawn using a fixed random seed (seed=42), yielding 120 subjects. The train/test split (80/20) was assigned at the **subject level**, stratified by age×gender, resulting in 96 train and 24 test subjects. All rows for a given subject fall in the same split, preventing data leakage.

### Data Construction

For each selected subject, all available visit rows were collected across all modalities. Visit rows with any missing KPI were dropped, retaining only baseline and 02_00_visit rows (04_00_visit and 06_00_visit had incomplete metabolite and CGM coverage). The final dataset contains 240 rows (120 subjects × 2 visits) with 661 columns: 6 KPIs, 640 tabular features (CGM 46, DEXA 294, retinal-cardiovascular proxy 128, metabolomics 172; DEXA features remain as inputs), and 13 metadata columns (IDs, demographics, split, modality flags).

Feature columns for each modality were assigned exclusively (no overlap) by source system. KPI columns were explicitly excluded from feature sets during modeling to prevent leakage.

### KPI Selection Rationale

KPIs were selected to represent clinically meaningful endpoints for each raw modality:

- **CGM:** `bt__hba1c` and `bt__glucose` were chosen over device-derived iglu metrics (GMI, TIR) because iglu metrics require a worn CGM sensor — fewer than 5 subjects had these at 2+ visits. Blood test markers are collected at every clinical visit and provide complete longitudinal coverage.
- **DEXA:** DEXA scan features (body composition + bone density) are included as **input features** for predicting KPIs from other modalities. DEXA KPIs (`total_scan_vat_mass`, `body_total_bmd`) are not prediction targets — DEXA is treated as an input modality only.
- **Retina (proxy):** No pre-computed retinal features (AVR, RNFL) exist in the dataset. `sitting_blood_pressure_systolic` and `intima_media_th_mm_1_intima_media_thickness` (carotid IMT) were used as cardiovascular proxies — both reflect the microvascular and atherosclerotic axes visible in fundus images.
- **Metabolomics:** `urate` (uric acid — metabolic syndrome, CVD risk) and `Bilirubin` (heme catabolism / liver oxidative stress) were selected from a stability analysis across all 133 metabolites. Population-level longitudinal stability (Pearson r between baseline and 02_00_visit, n=3,343) was r=0.60 for urate and r=0.56 for bilirubin — in the benchmark's "sweet spot" (r=0.5–0.75), where targets are predictable enough to be tractable but hard enough that a model must learn real signal. Higher-stability metabolites (r > 0.75, e.g. cortisol glucuronides, LysoPE species) were excluded as trivially easy longitudinal targets.

### Benchmark Tasks

**Task 1 — Cross-modal prediction:** For each of the 3 KPI-bearing modalities (CGM, retina, metabolites), its KPIs were held out and predicted from the **other 2 KPI-bearing modalities** at the same visit (DEXA features excluded from Task 1 inputs — 174–300 features depending on the held-out modality). KPI columns were excluded from all feature sets. This tests whether cross-modal information encodes the held-out modality's phenotype without any confounding from the rich DEXA feature set.

**Task 2 — Longitudinal prediction:** The **full feature set of all 4 modalities at baseline** (640 features total: CGM 46, DEXA 294, retina proxy 128, metabolomics 172) was used to predict all 6 KPIs at 02_00_visit (~2 years later). KPI columns from baseline were excluded as features. This tests the capacity to model biological trajectories across time.

### Models

Two models were evaluated for each task/KPI combination:

- **LightGBM (LGBM):** `n_estimators=100`, `learning_rate=0.05`, `max_depth=4`, `min_child_samples=5`. Missing values imputed with column medians prior to fitting. Feature names sanitized (special characters replaced with `_`) for LGBM compatibility.
- **Ridge regression:** sklearn `Pipeline` with median `SimpleImputer` → `StandardScaler` → `Ridge(alpha=10)`.

### Evaluation

**5-fold subject-level cross-validation** (random, `seed=42`). Each fold: 96 train subjects / 24 test subjects (all their visit rows used for Task 1; subject-level for Task 2). Primary metric: mean Pearson r ± std across 5 folds. Cells with fewer than 5 non-null test samples were skipped.

---

## Benchmark Results

### Task 1 — Cross-modal KPI prediction (missing modality)

Predict held-out modality's KPIs from the other 3 modalities at the **same visit**.
**5-fold CV**, ~192 train visits / ~48 test visits per fold.

Inputs: the other 2 KPI-bearing modalities only (DEXA features excluded). Feature counts: CGM held-out → 300 (retina+metabolites); Retina held-out → 218 (CGM+metabolites); Metabolites held-out → 174 (CGM+retina).

| Held-out | KPI | LGBM r (mean±std) | Ridge r (mean±std) | Interpretation |
|---|---|---|---|---|
| CGM | bt__hba1c | **0.602±0.268** | 0.028±0.052 | Strong (LGBM) — HbA1c partially encoded in metabolic+vascular state; Ridge fails without DEXA |
| CGM | bt__glucose | 0.089±0.169 | 0.127±0.082 | Poor — fasting glucose not recoverable cross-modally |
| Retina | sitting_BP_systolic | 0.180±0.200 | 0.138±0.190 | Weak — some vascular signal in CGM+metabolites, high variance |
| Retina | carotid_IMT | 0.162±0.234 | 0.102±0.104 | Poor — IMT needs direct ultrasound |
| Metabolites | urate | -0.101±0.268 | -0.161±0.192 | Poor — urate not recoverable from CGM+retina |
| Metabolites | Bilirubin | 0.016±0.193 | 0.053±0.256 | Poor — liver/oxidative state not cross-modally accessible |

### Task 2 — Longitudinal prediction (baseline → 02_00_visit)

Predict KPIs at **visit 2** given all modalities at **visit 1**.
**5-fold CV**, 120 subjects, ~96 train / ~24 test per fold.

| Modality | KPI | LGBM r (mean±std) | Ridge r (mean±std) | Interpretation |
|---|---|---|---|---|
| CGM | bt__hba1c | 0.296±0.261 | -0.067±0.242 | Weak — HbA1c fluctuates, high variance |
| CGM | bt__glucose | 0.209±0.268 | 0.211±0.064 | Weak — fasting glucose volatile |
| Retina | sitting_BP_systolic | **0.636±0.058** | **0.561±0.134** | Good — BP tracks reliably, consistent across folds |
| Retina | carotid_IMT | 0.301±0.102 | 0.323±0.194 | Moderate — slow atherosclerosis progression |
| Metabolites | urate | 0.481±0.148 | **0.515±0.143** | Moderate — urate longitudinally stable |
| Metabolites | Bilirubin | 0.233±0.146 | 0.284±0.184 | Weak — bilirubin fluctuates with lifestyle/diet |

---

## Interpretation

**What predicts well longitudinally (Task 2 mean r > 0.4):**
- BP systolic (0.636±0.058 / 0.561±0.134) — vascular tone tracks reliably; consistent across folds
- Urate (0.481±0.148 / 0.515±0.143) — moderately stable metabolite; Ridge slightly better
- Carotid IMT (0.301±0.102 / 0.323±0.194) — slow progression, moderate cross-fold variance

**What shows cross-modal signal (Task 1 mean r > 0.3):**
- HbA1c (0.602±0.268 LGBM) — glycemic state partially reflected in retina+metabolites; robust LGBM signal even without DEXA

**What doesn't predict (Task 1 mean r < 0.2):**
- Glucose, BP systolic, IMT, urate, Bilirubin — not recoverable from the other 2 KPI-bearing modalities
- This is the expected outcome: these KPIs require their native modality for reliable measurement

> **Note:** DEXA scan features (294 features: body composition + bone density) are included as inputs for **Task 2 only**. Task 1 uses only the 2 other KPI-bearing modalities as inputs. BMD and VAT are not prediction targets in either task.

**Interpretation for model evaluation:** A good multimodal model should achieve:
- Task 1 r >> baseline (e.g., HbA1c ≈ 0.60 with retina+metabolites only)
- Task 2 r >> baseline (e.g., BP ≈ 0.64, Urate ≈ 0.52, IMT ≈ 0.30)

---

## Key Limitations

1. **Retina KPIs are proxies** — No pre-computed AVR/RNFL. BP and IMT are the closest surrogates.
2. **Small test set per fold** — 24 test subjects per fold. Mean r is more reliable than any single split, but std can be large (especially for noisy KPIs like IMT, glucose).
3. **LGBM overfits on small n** — Negative mean r on urate/VAT/Bilirubin in Task 1. Ridge is more reliable when LGBM mean r < 0.
4. **Metabolite units** — Raw mass-spec abundances (arbitrary units), not calibrated concentrations.
5. **Dataset restricted to baseline + 02_00_visit** — 04_00_visit/06_00_visit dropped due to incomplete KPI coverage.

---

## Additional Eligible Subjects

**Stricter longitudinal criteria** (2+ visits in metabolomics, CGM, and retina — the core longitudinal modalities):

| Criterion | Count |
|---|---|
| Current benchmark | 120 subjects |
| Eligible pool (2+ visits in metab, CGM, retina + 1+ DEXA + 1+ blood) | 159 subjects |
| **New subjects available** | **156** |

This tighter pool supports modest expansion (+30%) but not doubling. The current 120 subject cohort represents 75% of all eligible subjects with complete longitudinal coverage.

### Flexible Modality Combinations

With relaxed criteria (any 1 picture + 1 timeseries + 1 tabular at 2+ visits), the pool expands dramatically:

| Rank | Combination | Subjects | New |
|---|---|---|---|
| 1 | retina + sleep + blood_test | 5,176 | 5,072 |
| 2 | retina + abi + blood_test | 5,082 | 4,978 |
| 3 | retina + sleep + microbiome | 4,911 | 4,808 |
| 4 | ultrasound + abi + blood_test | 4,716 | 4,608 |
| 5 | retina + abi + microbiome | 4,710 | 4,608 |
| 6 | ultrasound + abi + microbiome | 4,345 | 4,241 |
| 7 | ultrasound + sleep + blood_test | 4,265 | 4,157 |
| 8 | ultrasound + sleep + microbiome | 4,066 | 3,960 |
| 9 | retina + abi + nightingale | 2,787 | 2,739 |
| 10 | ultrasound + abi + nightingale | 2,765 | 2,714 |

### True Timeseries Signals (raw files)

| Modality | Type | Raw format |
|----------|------|------------|
| **CGM** | Timeseries | Glucose every ~15min (1341 pts/file) |
| **Sleep** | Timeseries | Itamar sleep data |
| **ECG** | Timeseries | 12-lead waveforms in text format (HR, intervals, voltage per lead) |
| **Voice** | Audio | .flac audio (~30s recordings) |
| **Gait** | Timeseries | Skeleton data (.ntds files) |

- **Picture**: retina, dexa, ultrasound
- **Timeseries**: cgm, sleep, ecg, gait, voice
- **Tabular**: metabolites, blood_test, microbiome, nightingale, proteomics, abi

This shows we could expand to 5000+ subjects if we switch modality combinations (e.g., using sleep instead of CGM).

---

## 2026 Expansion: 2-Visit vs 3-Visit (No Metabolomics)

We removed metabolomics and evaluated:
- `bench1_3mod`: microbiome + sleep + retina
- `bench2_5mod`: microbiome + sleep + retina + ultrasound + nightingale

### Cohort sizes

| Benchmark | Modalities | Window | Strict | Allow 1 Missing |
|---|---|---|---:|---:|
| bench1_3mod | microbiome + sleep + retina | baseline + 02_00_visit | 3,799 | 6,642 |
| bench1_3mod | microbiome + sleep + retina | baseline + 02_00_visit + 04_00_visit | 721 | 2,559 |
| bench2_5mod | microbiome + sleep + retina + ultrasound + nightingale | baseline + 02_00_visit | 200 | 4,184 |
| bench2_5mod | microbiome + sleep + retina + ultrasound + nightingale | baseline + 02_00_visit + 04_00_visit | 0 | 879 |

### Feature selection used for benchmarks (critical)

Top-2 KPI features selected per modality:

| Modality | Selected features |
|---|---|
| microbiome | `Rep_231`, `Rep_3541` |
| sleep | `total_valid_apnea_sleep_time`, `total_arousal_sleep_time` |
| retina proxy | `automorph_vein_average_width`, `automorph_artery_average_width` |
| ultrasound proxy | `r_abi`, `l_abi` |
| nightingale | `GlycA`, `VLDL_size` |

Model inputs:
- **Updated to full-X (current):** all feature columns from each modality system are used as X.
- Y remains the top-2 KPI targets listed above.
- Cross-modal task: all non-held-out modality features + `age`, `gender`, `bmi`.
- Longitudinal task: all source-visit modality features (excluding target column) + `age`, `gender`, `bmi`.

Retina feature definition (important):
- Retina now uses **retina-only automorph features** (`automorph_*`) from `cardiovascular_system`.
- Ultrasound proxy uses non-retina vascular columns (`abi/intima/carotid/plaque`) and excludes `automorph_*`.

### Benchmark subjects actually used in model runs

| Task | Benchmark | Window | Eligibility | Subjects used |
|---|---|---|---|---:|
| cross-modal | bench1_3mod | baseline + 02_00_visit | strict | 450 |
| cross-modal | bench1_3mod | baseline + 02_00_visit | allow1 | 1,831 |
| cross-modal | bench1_3mod | baseline + 02_00_visit + 04_00_visit | allow1 | 897 |
| cross-modal | bench2_5mod | baseline + 02_00_visit | allow1 | 1,038 |
| cross-modal | bench2_5mod | baseline + 02_00_visit + 04_00_visit | allow1 | 243 |
| longitudinal | bench1_3mod | baseline + 02_00_visit | strict | 450 |
| longitudinal | bench1_3mod | baseline + 02_00_visit | allow1 | 1,831 |
| longitudinal | bench1_3mod | baseline + 02_00_visit + 04_00_visit | allow1 | 897 |
| longitudinal | bench2_5mod | baseline + 02_00_visit | allow1 | 1,038 |
| longitudinal | bench2_5mod | baseline + 02_00_visit + 04_00_visit | allow1 | 243 |

### Benchmark performance (mean Pearson r across all KPI/fold rows)

| Task | Benchmark | Window | Eligibility | Ridge r | LGBM r |
|---|---|---|---|---:|---:|
| Missing modality (cross-modal) | bench1_3mod | 2 visits | allow1 | 0.045 | 0.088 |
| Missing modality (cross-modal) | bench1_3mod | 2 visits | strict | 0.042 | 0.073 |
| Missing modality (cross-modal) | bench1_3mod | 3 visits | allow1 | 0.049 | 0.068 |
| Missing modality (cross-modal) | bench2_5mod | 2 visits | allow1 | 0.088 | 0.172 |
| Missing modality (cross-modal) | bench2_5mod | 3 visits | allow1 | 0.063 | 0.142 |
| Longitudinal | bench1_3mod | 2 visits | allow1 | 0.222 | 0.354 |
| Longitudinal | bench1_3mod | 2 visits | strict | 0.216 | 0.367 |
| Longitudinal | bench1_3mod | 3 visits | allow1 | 0.147 | 0.311 |
| Longitudinal | bench2_5mod | 2 visits | allow1 | 0.137 | 0.288 |
| Longitudinal | bench2_5mod | 3 visits | allow1 | 0.129 | 0.203 |

Takeaways:
- Cross-modal prediction is modest once retina is constrained to retina-only automorph targets.
- Longitudinal prediction remains useful, with LGBM outperforming Ridge in the full-X setting.
- `bench2_5mod` remains allow1-only for stable evaluation in 3-visit mode.

---

## Files

| File | Description |
|---|---|
| `benchmark_subjects.csv` | Full dataset: 240 rows × 661 cols (all features + KPIs + split flag) |
| `benchmark_kpis_only.csv` | Compact: IDs + 6 KPIs + split + demographics |
| `benchmark_results.csv` | Model results for Task 1 and Task 2 |
| `benchmark_subjects_new_modalities_full_x.csv` | New full-X dataset for microbiome+sleep+retina(+ultrasound+nightingale), 3 visits |
| `29_benchmark_summary_2v3_full_x.csv` | Aggregate results for full-X 2-visit/3-visit experiments |
| `29_benchmark_results_2v3_full_x.csv` | Fold-level results for full-X experiments |
| `09_build_benchmark.py` | Subject selection + CSV construction |
| `10_run_benchmarks.py` | Benchmark task runner |
| `07_signal_overlap_alternatives.py` | Modality coverage analysis (led to choosing metabolites over proteomics) |
| `01_explore_modalities.py` | Initial modality exploration |
