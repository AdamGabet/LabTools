# Multimodal Baseline — Research Journal

## 2026-03-19 (update 2)

### Switched to 5-fold subject-level CV
- Replaced single train/test split (80/20) with 5-fold CV (random, seed=42, KFold)
- Each fold: 96 train / 24 test subjects. Task 1 uses all visit rows per subject; Task 2 uses subject-level (baseline → v2)
- Key CV findings vs previous single-split:
  - HbA1c Task 1: 0.68 → 0.593 (single split was slightly lucky)
  - Urate Task 2: 0.03 → 0.515 (single split badly underestimated — urate is actually moderate)
  - IMT Task 2: 0.50 → 0.323 (single split was lucky)
  - BMD and BP stable across both approaches (BMD: 0.98→0.941, BP: 0.63→0.636)
- CV is now the canonical evaluation; `benchmark_results.csv` updated with fold-level rows

---

## 2026-03-19

### What was built
- **Benchmark dataset**: 120 subjects × 2 visits = 240 rows, 100% KPI coverage on all rows
- **4 modalities**: CGM, DEXA, Retina (cardio proxies), Metabolites (metabolites_annotated)
- **8 KPIs**: bt__hba1c, bt__glucose, VAT, BMD, BP_systolic, IMT, urate, Tryptophan
- **Tasks**: Task 1 (cross-modal prediction) + Task 2 (baseline → 02_00_visit)
- **Files**: `09_build_benchmark.py`, `10_run_benchmarks.py`, `benchmark_subjects.csv`

### Key design decisions and why

**Why metabolites_annotated instead of proteomics:**
- Original plan was proteomics (CRP, CystatinC as KPIs)
- Old proteomics data had only 158 multi-visit subjects → replaced with metabolites_annotated (1,198)
- User added more proteomics data → now 953 multi-visit subjects (913 overlap with core3)
- BUT: proteomics at 02_00_visit is only 14.5% when intersected with sparse CGM blood tests
- Full eligibility check (all 8 KPIs at both visits): only 65 subjects → not viable for 120

**Why bt__hba1c/bt__glucose instead of iglu_gmi/iglu_in_range_70_180:**
- iglu metrics require worn CGM device → only ~4 subjects with 2+ visits
- Blood test markers available at every clinical visit → 46–47 subjects with 2+ visits

**Why subject-level train/test split:**
- CLAUDE.md requirement: always split at RegistrationCode level to prevent data leakage

**Why retina uses cardiovascular proxies:**
- Retina data = raw H5 images only (1024×1024 OD+OS), no pre-computed AVR/RNFL
- BP + carotid IMT from cardiovascular_system capture same microvascular axis

### Current open questions

1. **Metabolite KPIs updated** — switched Tryptophan → Bilirubin (liver/oxidative stress,
   population stability r=0.56). Bilirubin has better cross-modal signal (Task 1 Ridge r=0.39
   vs Tryptophan 0.19) because liver oxidative state is partially reflected in metabolic/
   vascular features. urate retained (r=0.60 population stability). See scripts 12 & 13.

2. **Can we switch to proteomics?** — Explored 2026-03-19:
   - Bottleneck: CGM blood tests sparse at 02_00_visit (~22–26%)
   - Any 4th modality with <30% v2 coverage will fail the full eligibility requirement
   - Nightingale NMR also fails (18% after CGM intersection → only 4 subjects)
   - **metabolites_annotated remains the only viable choice** unless we relax the
     CGM KPI requirement at 02_00_visit

3. **Fix metabolite KPI selection** — next step: run stability analysis across all
   metabolites_annotated features to find the most longitudinally stable pair
   with clinical relevance. See `11_explore_4th_modality.py` area for next script.

### Exploration script inventory

| Script | What it does | Key finding |
|---|---|---|
| 01 | Initial modality exploration | 1,518 subjects at baseline across all 4; 0 with 2+ visits if proteomics |
| 02–06 | Raw data / H5 structure checks | DEXA has images + tabular; retina images only |
| 07 | Signal overlap alternatives | metabolites_annotated has 1,198 multi-visit vs proteomics 158 |
| 08 | KPI candidate exploration | Chose BP+IMT as retina proxies; urate+Tryptophan for metabolites |
| 09 | Build benchmark CSV | 120 subjects, 240 rows, 100% KPI coverage |
| 10 | Run benchmarks | Task 1 + Task 2 results (see FINDINGS.md) |
| 11 | 4th modality candidate comparison | Proteomics/nightingale not viable; metabolites_annotated only option |
