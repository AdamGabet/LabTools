"""
Build 3-visit cohorts for new multimodal benchmarks.

Creates:
- row-level CSV with KPI values and modality-presence flags at each visit
- subject lists for strict and allow-1-missing criteria
"""

import json
from pathlib import Path

import pandas as pd

from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
IN_JSON = OUTDIR / "23_new_modality_candidates.json"
OUT_ROWS = OUTDIR / "24_three_visit_rows.csv"
OUT_SUBJECTS = OUTDIR / "24_three_visit_subjects.csv"

VISITS = ["baseline", "02_00_visit", "04_00_visit"]

SYSTEM_MAP = {
    "microbiome": "microbiome",
    "sleep": "sleep",
    "retina_proxy": "cardiovascular_system",
    "ultrasound_proxy": "cardiovascular_system",
    "nightingale": "nightingale",
}

BENCHMARKS = {
    "bench1_3mod": ["microbiome", "sleep", "retina_proxy"],
    "bench2_5mod": [
        "microbiome",
        "sleep",
        "retina_proxy",
        "ultrasound_proxy",
        "nightingale",
    ],
}


def build_base_rows(dfs):
    all_idx = set()
    for df in dfs.values():
        all_idx.update(set(df.index))
    rows = pd.DataFrame(sorted(all_idx), columns=["RegistrationCode", "research_stage"])
    rows = rows[rows["research_stage"].isin(VISITS)].copy()
    return rows.set_index(["RegistrationCode", "research_stage"])


def main():
    if not IN_JSON.exists():
        raise FileNotFoundError(f"Missing exploration JSON: {IN_JSON}")
    payload = json.loads(IN_JSON.read_text())
    chosen = payload["chosen_top2_kpis"]

    dfs = {sys_name: load_body_system_df(sys_name) for sys_name in set(SYSTEM_MAP.values())}
    rows = build_base_rows(dfs)

    # Add demographics for age filtering
    agb = load_body_system_df("Age_Gender_BMI")
    agb_base = (
        agb.xs("baseline", level="research_stage", drop_level=True)
        if "baseline" in agb.index.get_level_values("research_stage")
        else agb.groupby(level=0).first()
    )
    rows = rows.join(agb_base[["age", "gender", "bmi"]], how="left")
    rows = rows[(rows["age"] >= 40) & (rows["age"] <= 70)]

    # Add KPI columns and per-modality presence flags
    for mod, sys_name in SYSTEM_MAP.items():
        cols = chosen.get(mod, [])
        if len(cols) < 2:
            continue
        src = dfs[sys_name][cols].copy()
        new_names = {c: f"{mod}__{c}" for c in cols}
        src = src.rename(columns=new_names)
        rows = rows.join(src, how="left")
        renamed_cols = [new_names[c] for c in cols]
        rows[f"has_{mod}"] = rows[renamed_cols].notna().all(axis=1)

    # Keep only subjects with all 3 visits present in table
    visit_count = rows.reset_index().groupby("RegistrationCode")["research_stage"].nunique()
    keep_subjects = set(visit_count[visit_count == len(VISITS)].index)
    rows = rows.loc[rows.index.get_level_values(0).isin(keep_subjects)].copy()

    subject_rows = []
    flat = rows.reset_index()
    for bench_name, mods in BENCHMARKS.items():
        has_cols = [f"has_{m}" for m in mods if f"has_{m}" in flat.columns]
        tmp = flat[["RegistrationCode", "research_stage"] + has_cols].copy()
        tmp["n_present"] = tmp[has_cols].sum(axis=1)

        grp = tmp.groupby("RegistrationCode").agg(
            n_visits=("research_stage", "nunique"),
            min_present=("n_present", "min"),
        )
        strict = set(grp[(grp["n_visits"] == len(VISITS)) & (grp["min_present"] >= len(has_cols))].index)
        allow1 = set(
            grp[(grp["n_visits"] == len(VISITS)) & (grp["min_present"] >= max(1, len(has_cols) - 1))].index
        )

        for sid in sorted(set(grp.index)):
            subject_rows.append(
                {
                    "RegistrationCode": sid,
                    "benchmark": bench_name,
                    "n_modalities": len(has_cols),
                    "eligible_strict": sid in strict,
                    "eligible_allow1": sid in allow1,
                }
            )
        print(f"{bench_name}: strict={len(strict)}, allow1={len(allow1)}")

    rows.reset_index().to_csv(OUT_ROWS, index=False)
    pd.DataFrame(subject_rows).to_csv(OUT_SUBJECTS, index=False)
    print(f"Saved {OUT_ROWS}")
    print(f"Saved {OUT_SUBJECTS}")


if __name__ == "__main__":
    main()
