"""
Build an additional benchmark_subjects-style CSV for the new modality setup.

Output:
- benchmark_subjects_new_modalities.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
IN_ROWS = OUTDIR / "24_three_visit_rows.csv"
IN_SUBJECTS = OUTDIR / "24_three_visit_subjects.csv"
OUT_CSV = OUTDIR / "benchmark_subjects_new_modalities.csv"

SEED = 42
rng = np.random.default_rng(SEED)


def add_subject_split(df: pd.DataFrame) -> pd.DataFrame:
    subj = df[["RegistrationCode", "age", "gender"]].drop_duplicates("RegistrationCode").copy()
    subj = subj.dropna(subset=["age", "gender"])
    subj = subj[(subj["age"] >= 40) & (subj["age"] <= 70)]

    subj["age_bin"] = pd.cut(
        subj["age"],
        bins=[40, 50, 60, 70],
        labels=["40-50", "50-60", "60-70"],
        right=False,
    )
    subj["gender_bin"] = subj["gender"].map({0.0: "F", 1.0: "M", 0: "F", 1: "M"})
    subj = subj.dropna(subset=["age_bin", "gender_bin"])
    subj["split"] = "train"

    for age_bin in ["40-50", "50-60", "60-70"]:
        for gender_bin in ["F", "M"]:
            ids = subj[
                (subj["age_bin"] == age_bin) & (subj["gender_bin"] == gender_bin)
            ]["RegistrationCode"].tolist()
            if not ids:
                continue
            n_test = max(1, round(len(ids) * 0.2))
            chosen = rng.choice(ids, size=n_test, replace=False)
            subj.loc[subj["RegistrationCode"].isin(chosen), "split"] = "test"

    split_map = subj.set_index("RegistrationCode")["split"]
    df["split"] = df["RegistrationCode"].map(split_map).fillna("train")
    return df


def main():
    rows = pd.read_csv(IN_ROWS)
    subjects = pd.read_csv(IN_SUBJECTS)

    # Pivot benchmark eligibility flags to wide columns (one row per subject)
    wide = subjects.pivot_table(
        index="RegistrationCode",
        columns="benchmark",
        values=["eligible_strict", "eligible_allow1"],
        aggfunc="max",
        fill_value=False,
    )
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    out = rows.merge(wide, on="RegistrationCode", how="left")
    bool_cols = [c for c in out.columns if c.startswith("eligible_")]
    for c in bool_cols:
        out[c] = out[c].fillna(False).astype(bool)

    out = add_subject_split(out)

    # Keep consistent ordering
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
