"""
Summarize strict/allow1 cohort sizes for 2-visit and 3-visit windows.
"""

import json
from pathlib import Path

import pandas as pd

from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
IN_JSON = OUTDIR / "23_new_modality_candidates.json"
OUT_CSV = OUTDIR / "25_visit_count_summary.csv"

SYSTEM_MAP = {
    "microbiome": "microbiome",
    "sleep": "sleep",
    "retina_proxy": "cardiovascular_system",
    "ultrasound_proxy": "cardiovascular_system",
    "nightingale": "nightingale",
}

BENCHMARKS = {
    "bench1_3mod": ["microbiome", "sleep", "retina_proxy"],
    "bench2_5mod": ["microbiome", "sleep", "retina_proxy", "ultrasound_proxy", "nightingale"],
}

VISIT_WINDOWS = {
    "2visits_baseline_to_02": ["baseline", "02_00_visit"],
    "3visits_baseline_to_04": ["baseline", "02_00_visit", "04_00_visit"],
}


def main():
    payload = json.loads(IN_JSON.read_text())
    chosen = payload["chosen_top2_kpis"]
    dfs = {sys_name: load_body_system_df(sys_name) for sys_name in set(SYSTEM_MAP.values())}

    rows = []
    for mod, sys_name in SYSTEM_MAP.items():
        cols = chosen.get(mod, [])
        if len(cols) < 2:
            continue
        src = dfs[sys_name][cols].copy()
        for (sid, visit), vals in src.iterrows():
            rows.append(
                {
                    "RegistrationCode": sid,
                    "research_stage": visit,
                    "modality": mod,
                    "present": bool(vals.notna().all()),
                }
            )
    presence = pd.DataFrame(rows)

    out_rows = []
    for bench, mods in BENCHMARKS.items():
        sub = presence[presence["modality"].isin(mods)].copy()
        wide = (
            sub.pivot_table(
                index=["RegistrationCode", "research_stage"],
                columns="modality",
                values="present",
                aggfunc="max",
            )
            .fillna(False)
            .reset_index()
        )
        for window_name, visits in VISIT_WINDOWS.items():
            tmp = wide[wide["research_stage"].isin(visits)].copy()
            tmp["n_present"] = tmp[mods].sum(axis=1)
            grp = tmp.groupby("RegistrationCode").agg(
                n_visits=("research_stage", "nunique"),
                min_present=("n_present", "min"),
            )
            strict = int(((grp["n_visits"] == len(visits)) & (grp["min_present"] >= len(mods))).sum())
            allow1 = int(((grp["n_visits"] == len(visits)) & (grp["min_present"] >= len(mods) - 1)).sum())
            out_rows.append(
                {
                    "benchmark": bench,
                    "window": window_name,
                    "n_modalities": len(mods),
                    "strict": strict,
                    "allow1": allow1,
                }
            )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
