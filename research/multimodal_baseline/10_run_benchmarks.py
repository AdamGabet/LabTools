"""
Benchmark Tasks:
  Task 1: Given N-1 modalities at same visit → predict KPIs of the held-out modality
  Task 2: Given all modalities at visit 1   → predict all KPIs at visit 2

Models: LGBM, Ridge regression
Split:  5-fold subject-level cross-validation (random, seed=42)
"""

import sys, os

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

OUTDIR = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline"
N_FOLDS = 5
SEED = 42

# ── Load benchmark CSV ────────────────────────────────────────────────────
print("Loading benchmark data...")
df = pd.read_csv(os.path.join(OUTDIR, "benchmark_subjects.csv"))
print(f"Shape: {df.shape}")

# ── Feature / KPI definitions ─────────────────────────────────────────────
# DEXA is an input modality only — its KPIs are excluded from prediction targets
DEXA_KPIS = ["total_scan_vat_mass", "body_total_bmd"]

KPIS = {
    "cgm": ["bt__hba1c", "bt__glucose"],
    "retina": [
        "sitting_blood_pressure_systolic",
        "intima_media_th_mm_1_intima_media_thickness",
    ],
    "metabolites": [
        "urate",
        "Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin",
    ],
}
ALL_KPIS = [kpi for kpis in KPIS.values() for kpi in kpis]
EXCL_FROM_FEATS = ALL_KPIS + DEXA_KPIS  # columns never used as features

META_COLS = [
    "RegistrationCode",
    "research_stage",
    "split",
    "has_cgm",
    "has_dexa",
    "has_retina",
    "has_metabolites",
    "n_visits_cgm",
    "n_visits_dexa",
    "n_visits_metabolites",
]

print("Using age/gender/bmi as predictors (not metadata).")

all_mod_feat = [c for c in df.columns if c not in META_COLS + EXCL_FROM_FEATS]


def modality_feat_cols(mod_name, df_cols):
    if mod_name == "cgm":
        prefixes = ("iglu_", "bt__glucose", "bt__hba1c")
        return [
            c
            for c in df_cols
            if any(c.startswith(p) or c == p for p in prefixes)
            and c not in EXCL_FROM_FEATS
        ]
    elif mod_name == "dexa":
        prefixes = ("total_scan_", "body_comp_", "body_", "femur_", "spine_")
        return [
            c
            for c in df_cols
            if any(c.startswith(p) for p in prefixes) and c not in EXCL_FROM_FEATS
        ]
    elif mod_name == "retina":
        excl = set(META_COLS + EXCL_FROM_FEATS)
        cgm_f = set(modality_feat_cols("cgm", df_cols))
        dexa_f = set(modality_feat_cols("dexa", df_cols))
        met_f = set(modality_feat_cols("metabolites", df_cols))
        return [c for c in df_cols if c not in excl | cgm_f | dexa_f | met_f][:200]
    elif mod_name == "metabolites":
        excl = set(META_COLS + EXCL_FROM_FEATS)
        cgm_f = set(modality_feat_cols("cgm", df_cols))
        dexa_f = set(modality_feat_cols("dexa", df_cols))
        cardio_stub = [
            "from_",
            "l_brachial",
            "r_brachial",
            "l_ankle",
            "r_ankle",
            "lying_",
            "sitting_",
            "standing_",
            "intima_",
            "q_ms",
            "s_mv",
            "p_mv",
            "pr_ms",
            "qt_ms",
            "t_mv",
            "r_ms",
            "r_mv",
            "j_mv",
            "st_mv",
            "hr_bpm",
            "qrs_ms",
            "t_axis",
            "p_axis",
            "r_axis",
        ]
        met_cols = [
            c
            for c in df_cols
            if c not in excl | cgm_f | dexa_f
            and not any(c.startswith(s) for s in cardio_stub)
        ]
        return met_cols
    return []


MOD_FEAT = {m: modality_feat_cols(m, df.columns) for m in KPIS}
print("\nFeature counts per modality:")
for m, cols in MOD_FEAT.items():
    print(f"  {m}: {len(cols)} features")

# ── 5-fold subject-level splits ───────────────────────────────────────────
all_subjects = df["RegistrationCode"].unique()
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = [(train_idx, test_idx) for train_idx, test_idx in kf.split(all_subjects)]
print(f"\n5-fold CV: {N_FOLDS} folds over {len(all_subjects)} subjects")
for i, (tr, te) in enumerate(folds):
    print(f"  Fold {i + 1}: train={len(tr)} subjects, test={len(te)} subjects")


# ── Helpers ───────────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return {"n": len(y_true), "r": np.nan, "rmse": np.nan, "mae": np.nan}
    r, _ = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {
        "n": len(y_true),
        "r": round(r, 3),
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
    }


def make_models():
    return {
        "LGBM": LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_child_samples=5,
            n_jobs=4,
            random_state=42,
            verbose=-1,
        ),
        "Ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
    }


def sanitize_cols(df):
    import re

    mapping = {c: re.sub(r"[^A-Za-z0-9_]", "_", c) for c in df.columns}
    return df.rename(columns=mapping)


def run_prediction(X_train, y_train, X_test, y_test, kpi_name):
    results = []
    for mname, model in make_models().items():
        X_tr = sanitize_cols(X_train.copy()).fillna(X_train.median(numeric_only=True))
        X_te = sanitize_cols(X_test.copy()).fillna(X_train.median(numeric_only=True))
        mask_tr = y_train.notna()
        mask_te = y_test.notna()
        if mask_tr.sum() < 5 or mask_te.sum() < 5:
            continue
        try:
            model.fit(X_tr[mask_tr], y_train[mask_tr])
            pred = model.predict(X_te)
            metrics = eval_metrics(y_test.values, pred)
            metrics.update({"kpi": kpi_name, "model": mname})
            results.append(metrics)
        except Exception as e:
            results.append(
                {
                    "kpi": kpi_name,
                    "model": mname,
                    "n": 0,
                    "r": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "error": str(e),
                }
            )
    return results


# ─────────────────────────────────────────────────────────────────────────
# TASK 1: Cross-modal prediction — 5-fold CV
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 1: Cross-modal KPI prediction — 5-fold CV")
print("=" * 60)

task1_results = []

for held_out in KPIS.keys():  # cgm, retina, metabolites — DEXA excluded as target
    print(f"\n  Held-out: {held_out.upper()}")
    kpi_cols = KPIS[held_out]
    # DEXA features excluded from Task 1 inputs — only KPI-bearing modalities used
    input_feat = [
        c
        for mod, cols in MOD_FEAT.items()
        if mod != held_out and mod != "dexa"
        for c in cols
        if c in df.columns
    ]
    availability_col = f"has_{held_out}"
    task_df = df[df[availability_col].fillna(False)].copy()

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_subj = set(all_subjects[train_idx])
        test_subj = set(all_subjects[test_idx])
        train_df = task_df[task_df["RegistrationCode"].isin(train_subj)]
        test_df = task_df[task_df["RegistrationCode"].isin(test_subj)]

        X_train = train_df[input_feat].copy()
        X_test = test_df[input_feat].copy()

        for kpi in kpi_cols:
            if kpi not in df.columns:
                continue
            res = run_prediction(X_train, train_df[kpi], X_test, test_df[kpi], kpi)
            for r in res:
                r.update(
                    {
                        "task": "task1_cross_modal",
                        "held_out_modality": held_out,
                        "fold": fold_i + 1,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                    }
                )
            task1_results.extend(res)

    # Print per-KPI mean ± std across folds
    fold_df = pd.DataFrame(task1_results)
    fold_df = fold_df[fold_df["held_out_modality"] == held_out]
    for kpi in kpi_cols:
        kpi_fold = fold_df[fold_df["kpi"] == kpi]
        for mname in ["LGBM", "Ridge"]:
            rs = kpi_fold[kpi_fold["model"] == mname]["r"].dropna()
            if len(rs):
                print(f"    {kpi} | {mname}: r={rs.mean():.3f} ± {rs.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────
# TASK 2: Longitudinal prediction — 5-fold CV
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 2: Longitudinal KPI prediction — 5-fold CV")
print("=" * 60)

task2_results = []

v1, v2 = "baseline", "02_00_visit"
df_v1 = df[df["research_stage"] == v1].set_index("RegistrationCode")
df_v2 = df[df["research_stage"] == v2].set_index("RegistrationCode")
common = sorted(set(df_v1.index) & set(df_v2.index))
df_v1 = df_v1.loc[common]
df_v2 = df_v2.loc[common]
common_arr = np.array(common)
input_feat_all = [c for c in all_mod_feat if c in df_v1.columns]
X_all = df_v1[input_feat_all]

print(f"Subjects with both visits: {len(common)}")

# Build folds over subjects that have both visits
kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds2 = list(kf2.split(common_arr))

for fold_i, (train_idx, test_idx) in enumerate(folds2):
    train_subj = common_arr[train_idx]
    test_subj = common_arr[test_idx]
    X_train = X_all.loc[train_subj]
    X_test = X_all.loc[test_subj]

    for kpi in ALL_KPIS:
        if kpi not in df_v2.columns:
            continue
        y_train = df_v2.loc[train_subj, kpi]
        y_test = df_v2.loc[test_subj, kpi]
        mod = next(m for m, kpis in KPIS.items() if kpi in kpis)
        res = run_prediction(X_train, y_train, X_test, y_test, kpi)
        for r in res:
            r.update(
                {
                    "task": "task2_longitudinal",
                    "modality": mod,
                    "fold": fold_i + 1,
                    "v1": v1,
                    "v2": v2,
                    "n_train": len(train_subj),
                    "n_test": len(test_subj),
                }
            )
        task2_results.extend(res)

# Print per-KPI mean ± std
fold2_df = pd.DataFrame(task2_results)
for kpi in ALL_KPIS:
    kpi_fold = fold2_df[fold2_df["kpi"] == kpi]
    for mname in ["LGBM", "Ridge"]:
        rs = kpi_fold[kpi_fold["model"] == mname]["r"].dropna()
        if len(rs):
            print(f"  {kpi} | {mname}: r={rs.mean():.3f} ± {rs.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────
all_results = pd.DataFrame(task1_results + task2_results)
out_csv = os.path.join(OUTDIR, "benchmark_results.csv")
all_results.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# ── CV Summary tables (mean ± std r across 5 folds) ───────────────────────
print("\n" + "=" * 60)
print("TASK 1 CV SUMMARY — Cross-modal prediction (mean r ± std, 5 folds)")
print("=" * 60)
t1 = all_results[all_results["task"] == "task1_cross_modal"]
if len(t1) > 0:
    summary1 = (
        t1.groupby(["held_out_modality", "kpi", "model"])["r"]
        .agg(["mean", "std"])
        .round(3)
    )
    summary1.columns = ["r_mean", "r_std"]
    pivot1 = summary1.reset_index().pivot_table(
        index=["held_out_modality", "kpi"], columns="model", values=["r_mean", "r_std"]
    )
    # Print as mean ± std
    for (mod, kpi), row in summary1.reset_index().groupby(["held_out_modality", "kpi"]):
        parts = []
        for _, r in row.iterrows():
            parts.append(f"{r['model']}: {r['r_mean']:.3f}±{r['r_std']:.3f}")
        print(f"  {mod} | {kpi}: {' | '.join(parts)}")

print("\n" + "=" * 60)
print("TASK 2 CV SUMMARY — Longitudinal prediction (mean r ± std, 5 folds)")
print("=" * 60)
t2 = all_results[all_results["task"] == "task2_longitudinal"]
if len(t2) > 0:
    summary2 = (
        t2.groupby(["modality", "kpi", "model"])["r"].agg(["mean", "std"]).round(3)
    )
    summary2.columns = ["r_mean", "r_std"]
    for (mod, kpi), row in summary2.reset_index().groupby(["modality", "kpi"]):
        parts = []
        for _, r in row.iterrows():
            parts.append(f"{r['model']}: {r['r_mean']:.3f}±{r['r_std']:.3f}")
        print(f"  {mod} | {kpi}: {' | '.join(parts)}")
