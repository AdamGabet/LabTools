# Journal - multimodal_embedding_audit_20260407

## 2026-04-07

- Confirmed embedding directory contains `checkpoint.pt` and `embeddings.h5`.
- Inspected H5 schema:
  - `embeddings (240, 3, 256)`
  - `modality_present (240, 3)`
  - `targets (240, 6)`
  - visit metadata (`row_index`, `subject_index`, `research_stage`)
  - `meta.attrs` includes `d_model=256`, `n_visits=240`, training args.
- Reviewed upstream eval implementation in `eval/run_eval.py` to reproduce exact task structure and baseline comparison logic.
- Implemented `analysis.py` in this research folder to:
  - recompute Task1/Task2 probe metrics (subject-wise CV)
  - compute KPI deltas vs tabular baseline
  - run exploratory analyses: cross-modal retrieval, longitudinal subject fingerprint, and embedding-delta signal checks
  - generate figures and a PDF report.
- Ran analysis successfully and produced report + CSV artifacts.
- Wrote `FINDINGS.md` with interpretation and a novel retina-metabolites coupling observation.
- User requested explicit test with age/gender/bmi included as predictors in baseline benchmark.
- Updated `research/multimodal_baseline/10_run_benchmarks.py` so `age/gender/bmi` are not in metadata exclusion list.
- Re-ran `10_run_benchmarks.py` in uv environment (`source .venv/bin/activate.csh`) and regenerated `benchmark_results.csv`.
- Re-ran embedding audit against the new baseline and with AGB features appended to probe inputs.
- Added selected "cool" probes to report:
  - `iglu_cv`, `iglu_mage`, `iglu_hbgi`, `hr_bpm`, `qrs_ms`, `total_scan_sat_mass`, `body_comp_total_lean_mass`, `total_scan_vat_area`.
- Regenerated PDF report with updated baseline context + selected probe section.
