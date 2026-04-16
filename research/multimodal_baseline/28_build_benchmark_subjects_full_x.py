"""
Build benchmark subjects CSV with full feature lists as X.

Targets (Y) remain top-2 KPI columns per modality from 23_new_modality_candidates.json.
Features (X) include full modality feature sets:
- microbiome: all microbiome columns
- sleep: all sleep columns
- retina_proxy: cardiovascular columns excluding ultrasound-like terms
- ultrasound_proxy: cardiovascular columns with ultrasound-like terms
- nightingale: all nightingale columns
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
IN_JSON = OUTDIR / "30_stronger_targets.json"
FALLBACK_JSON = OUTDIR / "23_new_modality_candidates.json"
OUT_CSV = OUTDIR / "benchmark_subjects_new_modalities_full_x.csv"

VISITS = ["baseline", "02_00_visit", "04_00_visit"]
SEED = 42
rng = np.random.default_rng(SEED)


def split_cardio_columns(cardio_cols):
    retina = [c for c in cardio_cols if c.lower().startswith("automorph_")]
    ultra_terms = ("abi", "intima", "carotid", "plaque")
    ultra = [c for c in cardio_cols if c not in retina and any(t in c.lower() for t in ultra_terms)]
    return retina, ultra


def add_subject_split(df: pd.DataFrame) -> pd.DataFrame:
    subj = df[["RegistrationCode", "age", "gender"]].drop_duplicates("RegistrationCode").copy()
    subj = subj.dropna(subset=["age", "gender"])
    subj = subj[(subj["age"] >= 40) & (subj["age"] <= 70)]
    subj["age_bin"] = pd.cut(subj["age"], bins=[40, 50, 60, 70], labels=["40-50", "50-60", "60-70"], right=False)
    subj["gender_bin"] = subj["gender"].map({0.0: "F", 1.0: "M", 0: "F", 1: "M"})
    subj = subj.dropna(subset=["age_bin", "gender_bin"])
    subj["split"] = "train"
    for age_bin in ["40-50", "50-60", "60-70"]:
        for gender_bin in ["F", "M"]:
            ids = subj[(subj["age_bin"] == age_bin) & (subj["gender_bin"] == gender_bin)]["RegistrationCode"].tolist()
            if not ids:
                continue
            n_test = max(1, round(len(ids) * 0.2))
            chosen = rng.choice(ids, size=n_test, replace=False)
            subj.loc[subj["RegistrationCode"].isin(chosen), "split"] = "test"
    df["split"] = df["RegistrationCode"].map(subj.set_index("RegistrationCode")["split"]).fillna("train")
    return df


def main():
    src = IN_JSON if IN_JSON.exists() else FALLBACK_JSON
    payload = json.loads(src.read_text())
    chosen = payload["chosen_top2_kpis"]

    micro = load_body_system_df("microbiome")
    sleep = load_body_system_df("sleep")
    cardio = load_body_system_df("cardiovascular_system")
    night = load_body_system_df("nightingale")
    agb = load_body_system_df("Age_Gender_BMI")

    retina_cols, ultra_cols = split_cardio_columns(list(cardio.columns))

    all_idx = set(micro.index) | set(sleep.index) | set(cardio.index) | set(night.index)
    rows = pd.DataFrame(sorted(all_idx), columns=["RegistrationCode", "research_stage"])
    rows = rows[rows["research_stage"].isin(VISITS)].copy()
    rows = rows.set_index(["RegistrationCode", "research_stage"])

    # demographics
    agb_base = (
        agb.xs("baseline", level="research_stage", drop_level=True)
        if "baseline" in agb.index.get_level_values("research_stage")
        else agb.groupby(level=0).first()
    )
    rows = rows.join(agb_base[["age", "gender", "bmi"]], how="left")
    rows = rows[(rows["age"] >= 40) & (rows["age"] <= 70)]

    # full X blocks with prefixes
    rows = rows.join(micro.add_prefix("microbiome__"), how="left")
    rows = rows.join(sleep.add_prefix("sleep__"), how="left")
    rows = rows.join(cardio[retina_cols].add_prefix("retina_proxy__"), how="left")
    rows = rows.join(cardio[ultra_cols].add_prefix("ultrasound_proxy__"), how="left")
    rows = rows.join(night.add_prefix("nightingale__"), how="left")

    # modality presence based on target KPIs (top2)
    for mod, cols in chosen.items():
        pref = f"{mod}__"
        target_cols = [pref + c for c in cols]
        existing = [c for c in target_cols if c in rows.columns]
        rows[f"has_{mod}"] = rows[existing].notna().all(axis=1) if existing else False

    # keep subjects with all 3 visits represented in table
    vc = rows.reset_index().groupby("RegistrationCode")["research_stage"].nunique()
    keep = set(vc[vc == 3].index)
    rows = rows.loc[rows.index.get_level_values(0).isin(keep)].copy()

    # eligibility flags
    flat = rows.reset_index()
    benchmarks = {
        "bench1_3mod": ["microbiome", "sleep", "retina_proxy"],
        "bench2_5mod": ["microbiome", "sleep", "retina_proxy", "ultrasound_proxy", "nightingale"],
    }
    elig_rows = []
    for bench, mods in benchmarks.items():
        has_cols = [f"has_{m}" for m in mods]
        tmp = flat[["RegistrationCode", "research_stage"] + has_cols].copy()
        tmp["n_present"] = tmp[has_cols].sum(axis=1)
        grp = tmp.groupby("RegistrationCode").agg(n_visits=("research_stage", "nunique"), min_present=("n_present", "min"))
        strict = set(grp[(grp["n_visits"] == 3) & (grp["min_present"] >= len(mods))].index)
        allow1 = set(grp[(grp["n_visits"] == 3) & (grp["min_present"] >= len(mods) - 1)].index)
        for sid in grp.index:
            elig_rows.append(
                {
                    "RegistrationCode": sid,
                    f"eligible_strict__{bench}": sid in strict,
                    f"eligible_allow1__{bench}": sid in allow1,
                }
            )
    elig = pd.DataFrame(elig_rows).groupby("RegistrationCode", as_index=False).max()
    out = flat.merge(elig, on="RegistrationCode", how="left")
    out = add_subject_split(out)

    front = ["RegistrationCode", "research_stage", "split", "age", "gender", "bmi"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved {OUT_CSV}")
    print(f"Shape: {out.shape}")
    print(f"Subjects: {out['RegistrationCode'].nunique()}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
