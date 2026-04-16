# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-04-14

## User Preferences

- Short, clean code with meaningful comments. No narration comments ("# Import the module").
- One MD file per tool to explain code flow — be frugal creating them.
- Figures: `seaborn + matplotlib`, `plt.style.use('seaborn-v0_8-whitegrid')`, 300 DPI for PDFs. Always include at least one **explainability figure** (e.g., Feature Importance, SHAP, or a coefficient plot) to move beyond simple correlations.
- Palette: `#E74C3C` (red = increase), `#3498DB` (blue = decrease) — consistent throughout report.
- `report.pdf` is the primary deliverable of every study; a study is not done without it.
- **PDF links in chat:** Always give the **clickable HTTP URL** `http://127.0.0.1:8766/...` (research PDF server). Do **not** only give a repo path or `file://` — the user opens PDFs in the browser from that link.
- Always use `/home/adamgab/PycharmProjects/LabTools/.venv`; new deps via `uv pip install`.

## Key Learnings

- **Project:** LabTools — HPP 10K dataset research; N~5,000–10,000 subjects; cross-sectional; Israeli population.
- **Description:** Statistical analysis and ML tools for the Human Phenotype Project (HPP / 10K).
- **Age distribution:** 94% of subjects aged 40–70; ALWAYS filter to this range for stability. Check n > 500 per 5-year bin before any age analysis.
- **Subject ID:** `RegistrationCode` = `subject_id` = `10K_id`; format `"10K_" + 10 digits`; always split at subject level via `ids_folds`.
- **Correlation method:** Default to **Pearson**. Use Spearman only when distributions are clearly non-normal. Always document the choice and rationale.
- **Literature-first hypothesis:** Before correlating everything, search for recent (2024–2026) multi-omics trends (e.g., "functional modules" like SIRT2-Dipeptide axis). Use `partner` agent for this.
- **Data file names:** Verify the exact CSV file name in `BODY_SYSTEMS` before loading — e.g., `metabolites_annotated.csv` vs `metabolites.csv` can differ from what you expect.
- **Visual audit loop:** LLM cannot "see" PDFs. Use PyMuPDF (`fitz`) to render PDF pages → PNG, then feed images back for visual QC. Do this at the start, not as an afterthought.
- **Memory init:** At study start, initialise `journal.md` AND set up `generate_report.py` with the visual audit loop early — this gives "right-first-time" PDF delivery.
- **OpenCode ensemble:** `labresearch` (executor/PDF), `partner` (web), `planner` (plan + critic). OpenCode hooks do NOT run — manually update `.wolf/memory.md` after significant actions.

## Do-Not-Repeat

- [2026-04-14] Do NOT use `python -c "..."` for multi-line code — triggers security warnings. Write to a file instead.
- [2026-04-14] Do NOT use `train_test_split` on rows — always use `ids_folds` for subject-level splits. Data leakage is silent and fatal.
- [2026-04-14] Do NOT skip the visual audit loop until the end — render PDF → PNG early so figure quality can be iterated on.
- [2026-04-14] Do NOT assume the metabolomics file is named `metabolites.csv` — check the actual file name in `BODY_SYSTEMS` via `BiomarkerBrowser` or directory listing first.
- [2026-04-14] Do NOT declare a study done without `report.pdf` existing and reflecting the latest results.

## Decision Log

- [2026-04-14] OpenCode agent ensemble: `labresearch` (primary, owns venv + PDF), `partner` (web only), `planner` (plan + critique). Chosen over single-agent approach for separation of concerns and cleaner permission scoping.
- [2026-04-14] OpenWolf hooks bridged to OpenCode via `.opencode/plugins/openwolf.mjs` — shells out to existing `node .wolf/hooks/*.js` scripts with `CLAUDE_PROJECT_DIR` set. Auto-loaded from `.opencode/plugins/`; no entry needed in `opencode.json`.
- [2026-04-14] Pearson preferred over Spearman by default. Spearman is an option when data is clearly non-normal, but the choice is left to agent discretion based on the research. Always state which was used and why.
- [2026-04-14] `opencode.json` `instructions` field loads `CLAUDE.md` + `.wolf/OPENWOLF.md` globally; agent prompts reference `.claude/*.md` skills on-demand rather than loading all at once (token efficiency).
