"""
Explore KPI candidates and 3-visit feasibility for requested modality benchmarks.

Current benchmarks:
1) microbiome + sleep + retina
2) add ultrasound + nightingale
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_system_loader.biomarker_browser import BiomarkerBrowser
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
OUT_JSON = OUTDIR / "23_new_modality_candidates.json"
OUT_MD = OUTDIR / "23_new_modality_candidates.md"
OVERLAP_CSV = OUTDIR / "subject_test_date_overlap.csv"

# Map benchmark modalities to available tabular systems
SYSTEM_MAP = {
    "microbiome": "microbiome",
    "sleep": "sleep",
    # retina and ultrasound are image/raw modalities; use clinical proxies in cardiovascular_system
    "retina_proxy": "cardiovascular_system",
    "ultrasound_proxy": "cardiovascular_system",
    "nightingale": "nightingale",
}

VISITS = ["baseline", "02_00_visit", "04_00_visit"]
MAX_PER_MODALITY = 6


def top_numeric_candidates(
    df: pd.DataFrame,
    prefer_terms=None,
    exclude_terms=None,
    include_terms=None,
):
    prefer_terms = prefer_terms or []
    exclude_terms = exclude_terms or []
    rows = []
    n_total = len(df)
    stage_counts = df.index.get_level_values("research_stage").value_counts().to_dict()
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        n = int(s.notna().sum())
        if n < 200:
            continue
        std = float(s.std(skipna=True)) if n else np.nan
        if not np.isfinite(std) or std <= 0:
            continue
        cv = float(std / (abs(float(s.mean(skipna=True))) + 1e-9))
        name_l = col.lower()
        if any(t in name_l for t in exclude_terms):
            continue
        if include_terms and not any(t in name_l for t in include_terms):
            continue
        boost = 1 if any(t in name_l for t in prefer_terms) else 0
        rows.append(
            {
                "column": col,
                "non_null_n": n,
                "non_null_pct": round(100 * n / max(n_total, 1), 1),
                "std": round(std, 4),
                "cv": round(cv, 4),
                "term_boost": boost,
                "stage_counts": stage_counts,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return []
    out = out.sort_values(
        by=["term_boost", "non_null_n", "std", "cv"], ascending=[False, False, False, False]
    )
    return out.head(MAX_PER_MODALITY).to_dict(orient="records")


def visit_coverage(df: pd.DataFrame, cols):
    cov = {}
    for v in VISITS:
        if v not in df.index.get_level_values("research_stage"):
            cov[v] = {"subjects_any": 0, "subjects_all_cols": 0}
            continue
        xv = df.xs(v, level="research_stage")
        sub_any = int(xv[cols].notna().any(axis=1).sum())
        sub_all = int(xv[cols].notna().all(axis=1).sum())
        cov[v] = {"subjects_any": sub_any, "subjects_all_cols": sub_all}
    return cov


def pick_top2(candidates):
    return [c["column"] for c in candidates[:2]]


def modality_presence_by_visit(subjects, systems, chosen_kpis):
    """
    Build per-subject per-visit modality presence matrix:
    present if both chosen KPIs for that modality are non-null at that visit.
    """
    rows = []
    for mod, sys_name in systems.items():
        df = load_body_system_df(sys_name)
        cols = chosen_kpis.get(mod, [])
        if len(cols) < 2:
            continue
        for v in VISITS:
            if v not in df.index.get_level_values("research_stage"):
                continue
            xv = df.xs(v, level="research_stage")
            xv = xv.loc[xv.index.isin(subjects)]
            present = xv[cols].notna().all(axis=1)
            for sid, is_ok in present.items():
                rows.append(
                    {
                        "RegistrationCode": sid,
                        "research_stage": v,
                        "modality": mod,
                        "present": bool(is_ok),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def main():
    browser = BiomarkerBrowser()
    systems_summary = browser.get_system_summary()

    if not OVERLAP_CSV.exists():
        raise FileNotFoundError(f"Missing overlap CSV: {OVERLAP_CSV}")
    overlap = pd.read_csv(OVERLAP_CSV)

    overlap_counts = {
        "microbiome+sleep+retina": int(
            (
                (overlap["n_dates_microbiome"] > 0)
                & (overlap["n_dates_sleep"] > 0)
                & (overlap["n_dates_retina"] > 0)
            ).sum()
        ),
        "microbiome+sleep+retina+ultrasound+nightingale": int(
            (
                (overlap["n_dates_microbiome"] > 0)
                & (overlap["n_dates_sleep"] > 0)
                & (overlap["n_dates_retina"] > 0)
                & (overlap["n_dates_ultrasound"] > 0)
                & (overlap["n_dates_nightingale"] > 0)
            ).sum()
        ),
        "microbiome+sleep+retina (>=3 dates each)": int(
            (
                (overlap["n_dates_microbiome"] >= 3)
                & (overlap["n_dates_sleep"] >= 3)
                & (overlap["n_dates_retina"] >= 3)
            ).sum()
        ),
        "microbiome+sleep+retina+ultrasound+nightingale (>=3 dates each)": int(
            (
                (overlap["n_dates_microbiome"] >= 3)
                & (overlap["n_dates_sleep"] >= 3)
                & (overlap["n_dates_retina"] >= 3)
                & (overlap["n_dates_ultrasound"] >= 3)
                & (overlap["n_dates_nightingale"] >= 3)
            ).sum()
        ),
    }

    data = {}
    loaded = {}
    for mod, sys_name in SYSTEM_MAP.items():
        if sys_name not in loaded:
            loaded[sys_name] = load_body_system_df(sys_name)
        df = loaded[sys_name]
        if mod == "microbiome":
            data[mod] = top_numeric_candidates(
                df,
                prefer_terms=["shannon", "simpson", "divers", "firmic", "bacteroid", "alpha"],
                exclude_terms=["_id", "subject"],
            )
        elif mod == "sleep":
            data[mod] = top_numeric_candidates(
                df,
                prefer_terms=["ahi", "sleep", "spo2", "oxygen", "snoring", "rem", "hr"],
                exclude_terms=["_id", "subject"],
            )
        elif mod == "retina_proxy":
            data[mod] = top_numeric_candidates(
                df,
                prefer_terms=["automorph", "vessel", "artery", "vein", "cdr", "disc", "cup", "tortuosity"],
                include_terms=["automorph_"],
            )
        elif mod == "ultrasound_proxy":
            data[mod] = top_numeric_candidates(
                df,
                prefer_terms=["intima", "media", "carotid", "plaque", "abi"],
                exclude_terms=["automorph_"],
            )
        elif mod == "nightingale":
            data[mod] = top_numeric_candidates(
                df,
                prefer_terms=[
                    "hdl",
                    "ldl",
                    "glyca",
                    "triglycer",
                    "apob",
                    "omega",
                    "cholesterol",
                ],
            )

    # Coverage check for top-2 per modality
    coverage = {}
    chosen_kpis = {}
    for mod, candidates in data.items():
        cols = [c["column"] for c in candidates[:2]]
        chosen_kpis[mod] = cols
        if not cols:
            coverage[mod] = {}
            continue
        coverage[mod] = visit_coverage(loaded[SYSTEM_MAP[mod]], cols)

    # 3-visit missing-modality feasibility
    bench1 = ["microbiome", "sleep", "retina_proxy"]
    bench2 = bench1 + ["ultrasound_proxy", "nightingale"]
    subjects = set(overlap["registration_code"].astype(str))
    presence = modality_presence_by_visit(subjects, SYSTEM_MAP, chosen_kpis)

    feasibility = {}
    if not presence.empty:
        wide = (
            presence.pivot_table(
                index=["RegistrationCode", "research_stage"],
                columns="modality",
                values="present",
                aggfunc="max",
            )
            .fillna(False)
            .reset_index()
        )
        for mods_name, mods in [("bench1_3mod", bench1), ("bench2_5mod", bench2)]:
            keep_cols = [m for m in mods if m in wide.columns]
            tmp = wide[["RegistrationCode", "research_stage"] + keep_cols].copy()
            tmp["n_present"] = tmp[keep_cols].sum(axis=1)
            subj_stats = (
                tmp[tmp["research_stage"].isin(VISITS)]
                .groupby("RegistrationCode")
                .agg(
                    n_visits=("research_stage", "nunique"),
                    min_present=("n_present", "min"),
                )
            )
            strict = int(
                ((subj_stats["n_visits"] == len(VISITS)) & (subj_stats["min_present"] >= len(keep_cols))).sum()
            )
            allow_missing_1 = int(
                ((subj_stats["n_visits"] == len(VISITS)) & (subj_stats["min_present"] >= len(keep_cols) - 1)).sum()
            )
            feasibility[mods_name] = {
                "modalities": keep_cols,
                "subjects_3visits_strict": strict,
                "subjects_3visits_allow_1_missing_modality": allow_missing_1,
            }

    payload = {
        "systems_summary": systems_summary,
        "overlap_counts": overlap_counts,
        "candidate_kpis": data,
        "chosen_top2_kpis": chosen_kpis,
        "top2_visit_coverage": coverage,
        "three_visit_feasibility": feasibility,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# New Modality Candidate Exploration",
        "",
        "## Subject overlap counts",
        f"- 3-mod (microbiome+sleep+retina): {overlap_counts['microbiome+sleep+retina']}",
        (
            "- 5-mod (+ultrasound+nightingale): "
            f"{overlap_counts['microbiome+sleep+retina+ultrasound+nightingale']}"
        ),
        (
            "- 3-mod with >=3 dates each: "
            f"{overlap_counts['microbiome+sleep+retina (>=3 dates each)']}"
        ),
        (
            "- 5-mod with >=3 dates each: "
            f"{overlap_counts['microbiome+sleep+retina+ultrasound+nightingale (>=3 dates each)']}"
        ),
        "",
        "## Top KPI candidates by modality",
    ]
    for mod, candidates in data.items():
        md_lines.append(f"### {mod}")
        if not candidates:
            md_lines.append("- No candidates found")
            continue
        for c in candidates:
            md_lines.append(
                f"- `{c['column']}` | n={c['non_null_n']} ({c['non_null_pct']}%), std={c['std']}, cv={c['cv']}"
            )
        cov = coverage.get(mod, {})
        if cov:
            md_lines.append(
                f"- top2 coverage baseline(all): {cov['baseline']['subjects_all_cols']}, "
                f"02_00(all): {cov['02_00_visit']['subjects_all_cols']}, "
                f"04_00(all): {cov['04_00_visit']['subjects_all_cols']}"
            )
        md_lines.append("")

    md_lines.extend(
        [
            "## 3-Visit Feasibility (using top2 KPIs/modality)",
            f"- bench1 strict (3/3 modalities at all 3 visits): {feasibility.get('bench1_3mod', {}).get('subjects_3visits_strict', 0)}",
            (
                "- bench1 allow 1 missing modality per visit: "
                f"{feasibility.get('bench1_3mod', {}).get('subjects_3visits_allow_1_missing_modality', 0)}"
            ),
            f"- bench2 strict (5/5 modalities at all 3 visits): {feasibility.get('bench2_5mod', {}).get('subjects_3visits_strict', 0)}",
            (
                "- bench2 allow 1 missing modality per visit: "
                f"{feasibility.get('bench2_5mod', {}).get('subjects_3visits_allow_1_missing_modality', 0)}"
            ),
        ]
    )

    OUT_MD.write_text("\n".join(md_lines))
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
