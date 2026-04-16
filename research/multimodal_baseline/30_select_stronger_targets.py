"""
Select stronger benchmark targets (Y) per modality.

Criteria:
- high coverage at baseline/02_00_visit/04_00_visit
- longitudinal stability (baseline -> 02_00_visit Pearson r)
- avoid floor-effect dominated columns
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
OUT_JSON = OUTDIR / "30_stronger_targets.json"
OUT_MD = OUTDIR / "30_stronger_targets.md"

VISITS = ["baseline", "02_00_visit", "04_00_visit"]


def get_stage(df, stage):
    if stage not in df.index.get_level_values("research_stage"):
        return pd.DataFrame(index=pd.Index([], name="RegistrationCode"))
    return df.xs(stage, level="research_stage")


def pair_r(df, col, a="baseline", b="02_00_visit"):
    da = get_stage(df, a)[[col]]
    db = get_stage(df, b)[[col]]
    both = da.join(db, how="inner", lsuffix="_a", rsuffix="_b").dropna()
    if len(both) < 200:
        return np.nan, len(both)
    return float(both[f"{col}_a"].corr(both[f"{col}_b"])), len(both)


def floor_fraction(s):
    vals = s.dropna()
    if len(vals) == 0:
        return np.nan
    vmin = vals.min()
    return float((vals == vmin).mean())


def evaluate_columns(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        nn = int(s.notna().sum())
        if nn < 1000:
            continue
        cov = {}
        for v in VISITS:
            xv = get_stage(df, v)
            cov[v] = int(xv[c].notna().sum()) if c in xv.columns else 0
        r, n_pair = pair_r(df, c, "baseline", "02_00_visit")
        ff = floor_fraction(s)
        std = float(s.std(skipna=True))
        if not np.isfinite(std) or std <= 0:
            continue
        rows.append(
            {
                "column": c,
                "non_null_total": nn,
                "baseline_n": cov["baseline"],
                "v2_n": cov["02_00_visit"],
                "v4_n": cov["04_00_visit"],
                "r_base_v2": r,
                "n_pair_base_v2": n_pair,
                "floor_frac": ff,
                "std": std,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        out["r_base_v2"].fillna(-1) * 0.55
        + (out["v4_n"] / out["v4_n"].max()).fillna(0) * 0.25
        + (1 - out["floor_frac"].fillna(1)) * 0.20
    )
    return out.sort_values("score", ascending=False)


def main():
    micro = load_body_system_df("microbiome")
    sleep = load_body_system_df("sleep")
    cardio = load_body_system_df("cardiovascular_system")
    night = load_body_system_df("nightingale")

    retina_cols = [c for c in cardio.columns if c.lower().startswith("automorph_")]
    ultra_cols = [
        c
        for c in cardio.columns
        if (not c.lower().startswith("automorph_"))
        and any(t in c.lower() for t in ("abi", "intima", "carotid", "plaque"))
    ]

    evals = {
        "microbiome": evaluate_columns(micro, list(micro.columns)),
        "sleep": evaluate_columns(sleep, list(sleep.columns)),
        "retina_proxy": evaluate_columns(cardio, retina_cols),
        "ultrasound_proxy": evaluate_columns(cardio, ultra_cols),
        "nightingale": evaluate_columns(night, list(night.columns)),
    }

    chosen = {}
    for mod, t in evals.items():
        if t.empty:
            chosen[mod] = []
            continue
        # avoid selecting near-duplicate "same-family" pairs when possible
        picks = []
        used_tokens = set()
        for _, r in t.iterrows():
            name = r["column"]
            token = name.split("_")[0]
            if token in used_tokens and len(t) > 3:
                continue
            picks.append(name)
            used_tokens.add(token)
            if len(picks) == 2:
                break
        if len(picks) < 2:
            picks = t["column"].head(2).tolist()
        chosen[mod] = picks

    payload = {
        "selection_method": "stability_coverage_floor_score",
        "chosen_top2_kpis": chosen,
        "top_candidates": {k: v.head(15).to_dict(orient="records") for k, v in evals.items()},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines = ["# Stronger Target Selection", ""]
    for mod in ["microbiome", "sleep", "retina_proxy", "ultrasound_proxy", "nightingale"]:
        lines.append(f"## {mod}")
        lines.append(f"- chosen: {chosen.get(mod, [])}")
        t = evals[mod]
        if t.empty:
            lines.append("- no candidates")
            lines.append("")
            continue
        for _, r in t.head(5).iterrows():
            lines.append(
                f"- {r['column']} | r(base->v2)={r['r_base_v2']:.3f}, v4_n={int(r['v4_n'])}, floor={r['floor_frac']:.3f}, score={r['score']:.3f}"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_MD}")
    print("Chosen:", chosen)


if __name__ == "__main__":
    main()
