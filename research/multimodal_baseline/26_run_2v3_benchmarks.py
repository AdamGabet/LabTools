"""
Run 2-visit and 3-visit benchmarks for new modality stacks.

Tasks:
1) Missing-modality (cross-modal): predict held-out modality KPIs from other modalities at same visit
2) Longitudinal: predict next-visit KPIs from current-visit features
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
ROWS_CSV = OUTDIR / "24_three_visit_rows.csv"
KPI_JSON = OUTDIR / "23_new_modality_candidates.json"
OUT_RESULTS = OUTDIR / "26_benchmark_results_2v3.csv"
OUT_SUMMARY = OUTDIR / "26_benchmark_summary_2v3.csv"

BENCHMARKS = {
    "bench1_3mod": ["microbiome", "sleep", "retina_proxy"],
    "bench2_5mod": ["microbiome", "sleep", "retina_proxy", "ultrasound_proxy", "nightingale"],
}

WINDOWS = {
    "2visits_baseline_to_02": {
        "visits": ["baseline", "02_00_visit"],
        "pairs": [("baseline", "02_00_visit")],
    },
    "3visits_baseline_to_04": {
        "visits": ["baseline", "02_00_visit", "04_00_visit"],
        "pairs": [("baseline", "02_00_visit"), ("02_00_visit", "04_00_visit")],
    },
}


def metric_r(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 5:
        return np.nan, len(yt)
    return pearsonr(yt, yp)[0], len(yt)


def make_models():
    return {
        "Ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "LGBM": LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            min_child_samples=5,
            n_jobs=4,
            random_state=42,
            verbose=-1,
        ),
    }


def eligible_subjects(df, benchmark_mods, visits, mode):
    has_cols = [f"has_{m}" for m in benchmark_mods]
    tmp = df[df["research_stage"].isin(visits)][["RegistrationCode", "research_stage"] + has_cols].copy()
    tmp["n_present"] = tmp[has_cols].sum(axis=1)
    grp = tmp.groupby("RegistrationCode").agg(
        n_visits=("research_stage", "nunique"),
        min_present=("n_present", "min"),
    )
    if mode == "strict":
        keep = grp[(grp["n_visits"] == len(visits)) & (grp["min_present"] >= len(has_cols))]
    else:
        keep = grp[(grp["n_visits"] == len(visits)) & (grp["min_present"] >= len(has_cols) - 1)]
    return set(keep.index)


def get_kpi_cols(chosen_map, mods):
    out = {}
    for m in mods:
        out[m] = [f"{m}__{c}" for c in chosen_map[m]]
    return out


def run_cross_modal(df, bench_name, window_name, mode, mods, kpi_cols):
    subjects = sorted(eligible_subjects(df, mods, WINDOWS[window_name]["visits"], mode))
    if len(subjects) < 20:
        return []
    subdf = df[df["RegistrationCode"].isin(subjects) & df["research_stage"].isin(WINDOWS[window_name]["visits"])].copy()
    all_subjects = np.array(sorted(subdf["RegistrationCode"].unique()))
    n_splits = min(5, len(all_subjects))
    if n_splits < 2:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for held_out in mods:
        target_cols = kpi_cols[held_out]
        feat_cols = [c for m in mods if m != held_out for c in kpi_cols[m]]
        feat_cols = feat_cols + ["age", "gender", "bmi"]
        for fold, (tr, te) in enumerate(kf.split(all_subjects), start=1):
            train_subj = set(all_subjects[tr])
            test_subj = set(all_subjects[te])
            tr_df = subdf[subdf["RegistrationCode"].isin(train_subj)]
            te_df = subdf[subdf["RegistrationCode"].isin(test_subj)]
            for target in target_cols:
                # ensure target is present
                tr_use = tr_df[tr_df[target].notna()]
                te_use = te_df[te_df[target].notna()]
                if len(tr_use) < 20 or len(te_use) < 10:
                    continue
                Xtr = tr_use[feat_cols].copy()
                Xte = te_use[feat_cols].copy()
                ytr = tr_use[target].astype(float).values
                yte = te_use[target].astype(float).values

                for model_name, model in make_models().items():
                    try:
                        if model_name == "LGBM":
                            Xtr_fit = Xtr.fillna(Xtr.median(numeric_only=True))
                            Xte_fit = Xte.fillna(Xtr.median(numeric_only=True))
                        else:
                            Xtr_fit, Xte_fit = Xtr, Xte
                        model.fit(Xtr_fit, ytr)
                        pred = model.predict(Xte_fit)
                        r, n_eval = metric_r(yte, pred)
                        rows.append(
                            {
                                "task": "cross_modal_missing_modality",
                                "benchmark": bench_name,
                                "window": window_name,
                                "eligibility": mode,
                                "held_out_modality": held_out,
                                "target": target,
                                "model": model_name,
                                "fold": fold,
                                "n_subjects": len(subjects),
                                "n_train_rows": len(tr_use),
                                "n_test_rows": len(te_use),
                                "n_eval": n_eval,
                                "r": r,
                            }
                        )
                    except Exception:
                        continue
    return rows


def run_longitudinal(df, bench_name, window_name, mode, mods, kpi_cols):
    subjects = sorted(eligible_subjects(df, mods, WINDOWS[window_name]["visits"], mode))
    if len(subjects) < 20:
        return []
    subdf = df[df["RegistrationCode"].isin(subjects)].copy()
    all_subjects = np.array(sorted(subdf["RegistrationCode"].unique()))
    n_splits = min(5, len(all_subjects))
    if n_splits < 2:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    all_targets = [c for m in mods for c in kpi_cols[m]]
    base_feat_cols = all_targets + ["age", "gender", "bmi"]

    for src_visit, dst_visit in WINDOWS[window_name]["pairs"]:
        src = subdf[subdf["research_stage"] == src_visit].set_index("RegistrationCode")
        dst = subdf[subdf["research_stage"] == dst_visit].set_index("RegistrationCode")
        common = sorted(set(src.index) & set(dst.index))
        if len(common) < 20:
            continue
        src = src.loc[common]
        dst = dst.loc[common]
        subj_arr = np.array(common)
        for fold, (tr, te) in enumerate(kf.split(subj_arr), start=1):
            tr_subj = subj_arr[tr]
            te_subj = subj_arr[te]
            for target in all_targets:
                feat_cols = [c for c in base_feat_cols if c != target]
                Xtr = src.loc[tr_subj, feat_cols]
                Xte = src.loc[te_subj, feat_cols]
                ytr = dst.loc[tr_subj, target].astype(float)
                yte = dst.loc[te_subj, target].astype(float)
                mask_tr = ytr.notna()
                mask_te = yte.notna()
                if mask_tr.sum() < 20 or mask_te.sum() < 10:
                    continue

                for model_name, model in make_models().items():
                    try:
                        if model_name == "LGBM":
                            Xtr_fit = Xtr[mask_tr].fillna(Xtr[mask_tr].median(numeric_only=True))
                            Xte_fit = Xte[mask_te].fillna(Xtr[mask_tr].median(numeric_only=True))
                        else:
                            Xtr_fit = Xtr[mask_tr]
                            Xte_fit = Xte[mask_te]
                        model.fit(Xtr_fit, ytr[mask_tr].values)
                        pred = model.predict(Xte_fit)
                        r, n_eval = metric_r(yte[mask_te].values, pred)
                        rows.append(
                            {
                                "task": "longitudinal",
                                "benchmark": bench_name,
                                "window": window_name,
                                "eligibility": mode,
                                "transition": f"{src_visit}->{dst_visit}",
                                "target": target,
                                "model": model_name,
                                "fold": fold,
                                "n_subjects": len(subjects),
                                "n_eval": n_eval,
                                "r": r,
                            }
                        )
                    except Exception:
                        continue
    return rows


def main():
    df = pd.read_csv(ROWS_CSV)
    payload = json.loads(KPI_JSON.read_text())
    chosen = payload["chosen_top2_kpis"]
    all_rows = []

    for bench_name, mods in BENCHMARKS.items():
        kpi_cols = get_kpi_cols(chosen, mods)
        for window_name in WINDOWS:
            for mode in ["strict", "allow1"]:
                all_rows.extend(run_cross_modal(df, bench_name, window_name, mode, mods, kpi_cols))
                all_rows.extend(run_longitudinal(df, bench_name, window_name, mode, mods, kpi_cols))

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT_RESULTS, index=False)

    if len(res) == 0:
        print("No benchmark rows produced.")
        return

    summary = (
        res.groupby(["task", "benchmark", "window", "eligibility", "model"])["r"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "r_mean", "std": "r_std", "count": "n_rows"})
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {OUT_RESULTS}")
    print(f"Saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
