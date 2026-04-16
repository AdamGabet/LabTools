from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EMBEDDINGS_H5 = Path(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/adamgab/code_projects/multimodal_research_agent_c/embeddings/autoresearch_agent-c-20260407_215927/embeddings.h5"
)
BASELINE_CSV = Path(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_results.csv"
)
BENCHMARK_SUBJECTS_CSV = Path(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_subjects.csv"
)
MANIFEST_CSV = Path(
    "/home/adamgab/PycharmProjects/multimodal_research/data/interim/manifest.csv"
)
OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"

TARGET_COLUMNS = [
    "bt__hba1c",
    "bt__glucose",
    "sitting_blood_pressure_systolic",
    "intima_media_th_mm_1_intima_media_thickness",
    "urate",
    "Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin",
]

HELD_MODALITY_KPIS = {
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


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    out = np.empty(len(arr), dtype=object)
    for i, x in enumerate(arr):
        out[i] = x.decode("utf-8") if isinstance(x, bytes) else str(x)
    return out


def detect_modalities(n_mod: int) -> list[str]:
    if n_mod == 3:
        return ["cgm", "retina", "metabolites"]
    if n_mod == 4:
        return ["cgm", "dexa", "retina", "metabolites"]
    return [f"mod_{i}" for i in range(n_mod)]


def load_embeddings(path: Path) -> dict:
    with h5py.File(path, "r") as h5f:
        emb = np.asarray(h5f["embeddings"][:], dtype=np.float64)
        present = np.asarray(h5f["modality_present"][:], dtype=bool)
        targets = np.asarray(h5f["targets"][:], dtype=np.float64)
        row_index = np.asarray(h5f["row_index"][:], dtype=np.int32)
        subj = np.asarray(h5f["subject_index"][:], dtype=np.int32)
        stages = decode_str_array(np.asarray(h5f["research_stage"][:]))
        meta = dict(h5f["meta"].attrs)
    n_mod = emb.shape[1]
    return {
        "embeddings": emb,
        "modality_present": present,
        "targets": targets,
        "row_index": row_index,
        "subject_index": subj,
        "research_stage": stages,
        "meta": meta,
        "modalities": detect_modalities(n_mod),
    }


def load_demographics(data: dict) -> np.ndarray:
    manifest = pd.read_csv(
        MANIFEST_CSV, usecols=["row_index", "RegistrationCode", "research_stage"]
    )
    bench = pd.read_csv(
        BENCHMARK_SUBJECTS_CSV,
        usecols=["RegistrationCode", "research_stage", "age", "gender", "bmi"],
    )
    merged = manifest.merge(
        bench, on=["RegistrationCode", "research_stage"], how="left"
    )
    by_row = merged.set_index("row_index")[["age", "gender", "bmi"]]
    demo = by_row.reindex(data["row_index"]).to_numpy(dtype=np.float64)
    return demo


def ridge_fold_score(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    alpha: float = 10.0,
) -> float:
    mask_tr = np.isfinite(y_tr) & np.all(np.isfinite(x_tr), axis=1)
    mask_te = np.isfinite(y_te) & np.all(np.isfinite(x_te), axis=1)
    if mask_tr.sum() < 2 or mask_te.sum() < 2:
        return float("nan")
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(x_tr[mask_tr], y_tr[mask_tr])
    pred = model.predict(x_te[mask_te])
    if len(pred) < 2:
        return float("nan")
    r = pearsonr(y_te[mask_te], pred)[0]
    return float(r) if np.isfinite(r) else float("nan")


def baseline_best_r(baseline_df: pd.DataFrame, task: str, kpi: str) -> float:
    sub = baseline_df[(baseline_df["task"] == task) & (baseline_df["kpi"] == kpi)]
    if sub.empty:
        return float("nan")
    best = -np.inf
    for model_name in ["LGBM", "Ridge"]:
        rvals = pd.to_numeric(sub[sub["model"] == model_name]["r"], errors="coerce")
        mean_r = float(rvals.mean(skipna=True))
        if np.isfinite(mean_r) and mean_r > best:
            best = mean_r
    return best if np.isfinite(best) else float("nan")


def run_eval_like_probe(
    data: dict, baseline_df: pd.DataFrame, demographics: np.ndarray | None = None
) -> tuple[dict, pd.DataFrame]:
    emb = data["embeddings"]
    subj = data["subject_index"]
    y = data["targets"]
    stages = data["research_stage"]
    modality_names = data["modalities"]
    modality_to_idx = {m: i for i, m in enumerate(modality_names)}
    target_to_idx = {t: i for i, t in enumerate(TARGET_COLUMNS)}

    uniq_subjects = np.unique(subj)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    t1_scores = {k: [] for k in TARGET_COLUMNS}
    t2_scores = {k: [] for k in TARGET_COLUMNS}
    rows = []

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(uniq_subjects), start=1):
        tr_sub = set(uniq_subjects[tr_idx].tolist())
        te_sub = set(uniq_subjects[te_idx].tolist())
        tr_mask = np.array([s in tr_sub for s in subj])
        te_mask = np.array([s in te_sub for s in subj])

        for held_modality, kpis in HELD_MODALITY_KPIS.items():
            use_modalities = [
                m
                for m in HELD_MODALITY_KPIS
                if m != held_modality
                if m in modality_to_idx
            ]
            x_parts = [emb[:, modality_to_idx[m], :] for m in use_modalities]
            if demographics is not None:
                x_parts.append(demographics)
            x = np.concatenate(x_parts, axis=1)
            for kpi in kpis:
                k = target_to_idx[kpi]
                r = ridge_fold_score(
                    x[tr_mask], y[tr_mask, k], x[te_mask], y[te_mask, k]
                )
                t1_scores[kpi].append(r)
                rows.append(
                    {
                        "task": "task1_cross_modal",
                        "fold": fold_id,
                        "kpi": kpi,
                        "r": r,
                        "held_out_modality": held_modality,
                    }
                )

    by_sub = {}
    for i in range(len(subj)):
        by_sub.setdefault(int(subj[i]), {})[str(stages[i])] = i
    pairs = [
        (v["baseline"], v["02_00_visit"])
        for _, v in by_sub.items()
        if "baseline" in v and "02_00_visit" in v
    ]
    pairs = sorted(pairs, key=lambda p: subj[p[0]])

    x2_parts = [
        np.stack(
            [
                np.concatenate([emb[bl, m, :] for m in range(emb.shape[1])])
                for bl, _ in pairs
            ]
        )
    ]
    if demographics is not None:
        x2_parts.append(np.stack([demographics[bl] for bl, _ in pairs]))
    x2 = np.concatenate(x2_parts, axis=1)
    y2 = np.stack([y[v2, :] for _, v2 in pairs])
    pair_subj = np.array([subj[bl] for bl, _ in pairs])

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(uniq_subjects), start=1):
        tr_sub = set(uniq_subjects[tr_idx].tolist())
        te_sub = set(uniq_subjects[te_idx].tolist())
        tr_mask = np.array([s in tr_sub for s in pair_subj])
        te_mask = np.array([s in te_sub for s in pair_subj])
        for kpi_i, kpi in enumerate(TARGET_COLUMNS):
            r = ridge_fold_score(
                x2[tr_mask], y2[tr_mask, kpi_i], x2[te_mask], y2[te_mask, kpi_i]
            )
            t2_scores[kpi].append(r)
            rows.append(
                {
                    "task": "task2_longitudinal",
                    "fold": fold_id,
                    "kpi": kpi,
                    "r": r,
                    "held_out_modality": "",
                }
            )

    deltas = []
    score_table = []
    for kpi in TARGET_COLUMNS:
        t1 = float(np.nanmean(t1_scores[kpi]))
        t2 = float(np.nanmean(t2_scores[kpi]))
        b1 = baseline_best_r(baseline_df, "task1_cross_modal", kpi)
        b2 = baseline_best_r(baseline_df, "task2_longitudinal", kpi)
        d1 = t1 - b1 if np.isfinite(b1) else float("nan")
        d2 = t2 - b2 if np.isfinite(b2) else float("nan")
        if np.isfinite(d1):
            deltas.append(d1)
        if np.isfinite(d2):
            deltas.append(d2)
        score_table.append(
            {
                "task": "task1_cross_modal",
                "kpi": kpi,
                "mean_r": t1,
                "baseline_best_r": b1,
                "delta": d1,
            }
        )
        score_table.append(
            {
                "task": "task2_longitudinal",
                "kpi": kpi,
                "mean_r": t2,
                "baseline_best_r": b2,
                "delta": d2,
            }
        )

    metrics = {
        "eval_score": float(np.nanmean(deltas)) if deltas else float("nan"),
        "task1_mean_r": float(
            np.nanmean([np.nanmean(t1_scores[k]) for k in TARGET_COLUMNS])
        ),
        "task2_mean_r": float(
            np.nanmean([np.nanmean(t2_scores[k]) for k in TARGET_COLUMNS])
        ),
        "n_subjects": int(len(uniq_subjects)),
        "n_visits": int(len(subj)),
        "n_longitudinal_pairs": int(len(pairs)),
    }
    return metrics, pd.DataFrame(score_table), pd.DataFrame(rows)


def selected_feature_probes(data: dict, demographics: np.ndarray) -> pd.DataFrame:
    manifest = pd.read_csv(
        MANIFEST_CSV, usecols=["row_index", "RegistrationCode", "research_stage"]
    )
    bench = pd.read_csv(BENCHMARK_SUBJECTS_CSV)
    emb = data["embeddings"]
    subj = data["subject_index"]
    uniq_subjects = np.unique(subj)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    feature_specs = [
        ("iglu_cv", "cgm variability"),
        ("iglu_mage", "cgm excursions"),
        ("iglu_hbgi", "cgm hyper risk"),
        ("hr_bpm", "cardio resting rate"),
        ("qrs_ms", "cardio electrical"),
        ("total_scan_sat_mass", "dexa adiposity"),
        ("body_comp_total_lean_mass", "dexa lean mass"),
        ("total_scan_vat_area", "dexa visceral fat"),
    ]

    x_emb = np.concatenate([emb[:, i, :] for i in range(emb.shape[1])], axis=1)
    x_emb_agb = np.concatenate([x_emb, demographics], axis=1)
    row_meta = manifest.set_index("row_index").reindex(data["row_index"])
    bench_idx = bench.set_index(["RegistrationCode", "research_stage"])

    rows = []
    for feature_name, domain in feature_specs:
        y = np.full(len(subj), np.nan)
        for i in range(len(subj)):
            sid = row_meta.iloc[i]["RegistrationCode"]
            st = row_meta.iloc[i]["research_stage"]
            if (sid, st) in bench_idx.index and feature_name in bench_idx.columns:
                val = bench_idx.loc[(sid, st), feature_name]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                y[i] = pd.to_numeric(val, errors="coerce")

        emb_scores = []
        emb_agb_scores = []
        for tr_idx, te_idx in kf.split(uniq_subjects):
            tr_sub = set(uniq_subjects[tr_idx].tolist())
            te_sub = set(uniq_subjects[te_idx].tolist())
            tr_mask = np.array([s in tr_sub for s in subj])
            te_mask = np.array([s in te_sub for s in subj])
            emb_scores.append(
                ridge_fold_score(
                    x_emb[tr_mask], y[tr_mask], x_emb[te_mask], y[te_mask], alpha=10.0
                )
            )
            emb_agb_scores.append(
                ridge_fold_score(
                    x_emb_agb[tr_mask],
                    y[tr_mask],
                    x_emb_agb[te_mask],
                    y[te_mask],
                    alpha=10.0,
                )
            )

        rows.append(
            {
                "feature": feature_name,
                "domain": domain,
                "r_embedding": float(np.nanmean(emb_scores)),
                "r_embedding_plus_agb": float(np.nanmean(emb_agb_scores)),
                "gain_from_agb": float(
                    np.nanmean(emb_agb_scores) - np.nanmean(emb_scores)
                ),
                "n_non_missing": int(np.isfinite(y).sum()),
            }
        )
    return pd.DataFrame(rows)


def subject_fingerprint_accuracy(data: dict) -> pd.DataFrame:
    emb = data["embeddings"]
    subj = data["subject_index"]
    stage = data["research_stage"]
    mods = data["modalities"]

    baseline_idx = np.where(stage == "baseline")[0]
    follow_idx = np.where(stage == "02_00_visit")[0]
    b_map = {int(subj[i]): i for i in baseline_idx}
    f_map = {int(subj[i]): i for i in follow_idx}
    common = sorted(set(b_map) & set(f_map))
    if not common:
        return pd.DataFrame()

    b = np.array([b_map[s] for s in common], dtype=int)
    f = np.array([f_map[s] for s in common], dtype=int)
    n = len(common)
    rows = []
    for m_i, m_name in enumerate(mods):
        xb = emb[b, m_i, :]
        xf = emb[f, m_i, :]
        sim = cosine_similarity(xb, xf)
        pred = np.argmax(sim, axis=1)
        acc = float(np.mean(pred == np.arange(n)))
        rows.append(
            {
                "view": m_name,
                "top1_id_accuracy": acc,
                "chance": 1.0 / n,
                "n_subjects": n,
            }
        )

    xb = np.concatenate([emb[b, i, :] for i in range(emb.shape[1])], axis=1)
    xf = np.concatenate([emb[f, i, :] for i in range(emb.shape[1])], axis=1)
    sim = cosine_similarity(xb, xf)
    pred = np.argmax(sim, axis=1)
    rows.append(
        {
            "view": "concat_all",
            "top1_id_accuracy": float(np.mean(pred == np.arange(n))),
            "chance": 1.0 / n,
            "n_subjects": n,
        }
    )
    return pd.DataFrame(rows)


def cross_modal_retrieval(data: dict) -> pd.DataFrame:
    emb = data["embeddings"]
    mods = data["modalities"]
    rows = []
    n = emb.shape[0]
    for i, m_a in enumerate(mods):
        for j, m_b in enumerate(mods):
            if i == j:
                continue
            xa = emb[:, i, :]
            xb = emb[:, j, :]
            sim = cosine_similarity(xa, xb)
            pred = np.argmax(sim, axis=1)
            acc = float(np.mean(pred == np.arange(n)))
            rows.append(
                {
                    "from_modality": m_a,
                    "to_modality": m_b,
                    "top1_same_visit_accuracy": acc,
                    "chance": 1.0 / n,
                }
            )
    return pd.DataFrame(rows)


def longitudinal_delta_signal(data: dict) -> pd.DataFrame:
    emb = data["embeddings"]
    subj = data["subject_index"]
    stage = data["research_stage"]
    targets = data["targets"]
    mods = data["modalities"]

    baseline_idx = np.where(stage == "baseline")[0]
    follow_idx = np.where(stage == "02_00_visit")[0]
    b_map = {int(subj[i]): i for i in baseline_idx}
    f_map = {int(subj[i]): i for i in follow_idx}
    common = sorted(set(b_map) & set(f_map))
    if not common:
        return pd.DataFrame()

    b = np.array([b_map[s] for s in common], dtype=int)
    f = np.array([f_map[s] for s in common], dtype=int)
    rows = []
    hba1c_delta = targets[f, 0] - targets[b, 0]
    for m_i, m_name in enumerate(mods):
        delta_vec = emb[f, m_i, :] - emb[b, m_i, :]
        delta_norm = np.linalg.norm(delta_vec, axis=1)
        r = pearsonr(delta_norm, hba1c_delta)[0] if len(delta_norm) > 2 else np.nan
        rows.append({"modality": m_name, "corr_delta_norm_vs_hba1c_change": float(r)})

    cgm_idx = mods.index("cgm") if "cgm" in mods else 0
    met_idx = (
        mods.index("metabolites") if "metabolites" in mods else min(1, len(mods) - 1)
    )
    cgm_norm = np.linalg.norm(emb[f, cgm_idx, :] - emb[b, cgm_idx, :], axis=1)
    met_norm = np.linalg.norm(emb[f, met_idx, :] - emb[b, met_idx, :], axis=1)
    r_cross = pearsonr(cgm_norm, met_norm)[0] if len(cgm_norm) > 2 else np.nan
    rows.append(
        {
            "modality": "cgm_vs_metabolites",
            "corr_delta_norm_vs_hba1c_change": float(r_cross),
        }
    )
    return pd.DataFrame(rows)


def make_figures(
    data: dict,
    score_table: pd.DataFrame,
    fingerprint_df: pd.DataFrame,
    retrieval_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    selected_probe_df: pd.DataFrame,
) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    stage_counts = pd.Series(data["research_stage"]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    stage_counts.plot(kind="bar", ax=ax, color="#355070")
    ax.set_title("Visit counts by research stage")
    ax.set_ylabel("N visits")
    ax.set_xlabel("")
    fig.tight_layout()
    p = FIG_DIR / "stage_counts.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    piv = score_table.pivot(index="kpi", columns="task", values="delta").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(
                j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=8
            )
    ax.set_title("Probe delta vs tabular baseline (Pearson r)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="delta r")
    fig.tight_layout()
    p = FIG_DIR / "delta_heatmap.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    paths.append(p)

    if not fingerprint_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            fingerprint_df["view"], fingerprint_df["top1_id_accuracy"], color="#6d597a"
        )
        ax.axhline(
            float(fingerprint_df["chance"].iloc[0]),
            color="black",
            linestyle="--",
            linewidth=1,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Top-1 ID accuracy")
        ax.set_title("Subject fingerprint from baseline->follow-up")
        fig.tight_layout()
        p = FIG_DIR / "fingerprint_accuracy.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    if not retrieval_df.empty:
        mods = sorted(
            set(retrieval_df["from_modality"]) | set(retrieval_df["to_modality"])
        )
        mat = np.full((len(mods), len(mods)), np.nan)
        m2i = {m: i for i, m in enumerate(mods)}
        for _, row in retrieval_df.iterrows():
            mat[m2i[row["from_modality"]], m2i[row["to_modality"]]] = row[
                "top1_same_visit_accuracy"
            ]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.1, np.nanmax(mat)))
        ax.set_xticks(np.arange(len(mods)))
        ax.set_yticks(np.arange(len(mods)))
        ax.set_xticklabels(mods)
        ax.set_yticklabels(mods)
        for i in range(len(mods)):
            for j in range(len(mods)):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8
                    )
        ax.set_title("Cross-modal same-visit retrieval (top-1)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        p = FIG_DIR / "cross_modal_retrieval.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    if not delta_df.empty:
        d = delta_df[delta_df["modality"] != "cgm_vs_metabolites"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(d["modality"], d["corr_delta_norm_vs_hba1c_change"], color="#b56576")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel("Pearson r")
        ax.set_title("Association: embedding movement vs HbA1c change")
        fig.tight_layout()
        p = FIG_DIR / "delta_vs_hba1c.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    if not selected_probe_df.empty:
        dd = selected_probe_df.copy()
        fig, ax = plt.subplots(figsize=(8, 4.6))
        x = np.arange(len(dd))
        w = 0.38
        ax.bar(x - w / 2, dd["r_embedding"], w, label="embedding", color="#457b9d")
        ax.bar(
            x + w / 2,
            dd["r_embedding_plus_agb"],
            w,
            label="embedding+AGB",
            color="#e76f51",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(dd["feature"], rotation=25, ha="right")
        ax.set_ylabel("Mean Pearson r (5-fold subject CV)")
        ax.set_title("Selected novel probes: embeddings vs embeddings+AGB")
        ax.legend()
        fig.tight_layout()
        p = FIG_DIR / "selected_probes_comparison.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        paths.append(p)

    return paths


def build_pdf_report(
    fig_paths: list[Path],
    metrics: dict,
    summary_df: pd.DataFrame,
    metrics_demo: dict,
    selected_probe_df: pd.DataFrame,
    out_pdf: Path,
) -> None:
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            "Multimodal Embedding Audit",
            "",
            f"Embeddings file: {EMBEDDINGS_H5}",
            f"N visits: {metrics['n_visits']}",
            f"N subjects: {metrics['n_subjects']}",
            f"N baseline->02_00 pairs: {metrics['n_longitudinal_pairs']}",
            "",
            f"EVAL_SCORE (vs baseline): {metrics['eval_score']:.4f}",
            f"Task 1 mean r: {metrics['task1_mean_r']:.4f}",
            f"Task 2 mean r: {metrics['task2_mean_r']:.4f}",
            "",
            f"After rerun benchmark + AGB predictors: EVAL_SCORE={metrics_demo['eval_score']:.4f}",
            f"Task1={metrics_demo['task1_mean_r']:.4f}, Task2={metrics_demo['task2_mean_r']:.4f}",
            "",
            "Top KPI deltas (probe - baseline):",
        ]
        top = summary_df.sort_values("delta", ascending=False).head(6)
        for _, row in top.iterrows():
            lines.append(f"- {row['task']} | {row['kpi']}: {row['delta']:+.3f}")
        ax.text(0.03, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig)
        plt.close(fig)

        if not selected_probe_df.empty:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            txt = ["Selected novel probes (embedding+AGB - embedding):", ""]
            for _, r in selected_probe_df.sort_values(
                "r_embedding_plus_agb", ascending=False
            ).iterrows():
                txt.append(
                    f"- {r['feature']} ({r['domain']}): r={r['r_embedding_plus_agb']:.3f} "
                    f"(emb={r['r_embedding']:.3f}, gain={r['gain_from_agb']:+.3f}, n={int(r['n_non_missing'])})"
                )
            ax.text(0.04, 0.97, "\n".join(txt), va="top", ha="left", fontsize=10.5)
            pdf.savefig(fig)
            plt.close(fig)

        for p in fig_paths:
            img = plt.imread(p)
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = load_embeddings(EMBEDDINGS_H5)
    demographics = load_demographics(data)
    baseline_df = pd.read_csv(BASELINE_CSV)

    metrics, summary_df, fold_df = run_eval_like_probe(data, baseline_df)
    metrics_demo, summary_df_demo, fold_df_demo = run_eval_like_probe(
        data, baseline_df, demographics=demographics
    )
    selected_probe_df = selected_feature_probes(data, demographics)
    fingerprint_df = subject_fingerprint_accuracy(data)
    retrieval_df = cross_modal_retrieval(data)
    delta_df = longitudinal_delta_signal(data)

    summary_df.to_csv(OUT_DIR / "eval_vs_baseline_summary.csv", index=False)
    summary_df_demo.to_csv(
        OUT_DIR / "eval_vs_baseline_summary_with_agb.csv", index=False
    )
    fold_df.to_csv(OUT_DIR / "eval_fold_scores.csv", index=False)
    fold_df_demo.to_csv(OUT_DIR / "eval_fold_scores_with_agb.csv", index=False)
    fingerprint_df.to_csv(OUT_DIR / "subject_fingerprint.csv", index=False)
    retrieval_df.to_csv(OUT_DIR / "cross_modal_retrieval.csv", index=False)
    delta_df.to_csv(OUT_DIR / "longitudinal_delta_signal.csv", index=False)
    selected_probe_df.to_csv(OUT_DIR / "selected_feature_probes.csv", index=False)

    metrics_json = {
        k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
        for k, v in metrics.items()
    }
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    metrics_demo_json = {
        k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
        for k, v in metrics_demo.items()
    }
    with open(OUT_DIR / "metrics_with_agb.json", "w", encoding="utf-8") as f:
        json.dump(metrics_demo_json, f, indent=2)

    comparison = pd.DataFrame(
        {
            "metric": ["eval_score", "task1_mean_r", "task2_mean_r"],
            "embedding_only": [
                metrics_json["eval_score"],
                metrics_json["task1_mean_r"],
                metrics_json["task2_mean_r"],
            ],
            "embedding_plus_agb": [
                metrics_demo_json["eval_score"],
                metrics_demo_json["task1_mean_r"],
                metrics_demo_json["task2_mean_r"],
            ],
        }
    )
    comparison["delta"] = (
        comparison["embedding_plus_agb"] - comparison["embedding_only"]
    )
    comparison.to_csv(OUT_DIR / "metrics_comparison_agb.csv", index=False)

    fig_paths = make_figures(
        data,
        summary_df,
        fingerprint_df,
        retrieval_df,
        delta_df,
        selected_probe_df,
    )
    build_pdf_report(
        fig_paths,
        metrics,
        summary_df,
        metrics_demo,
        selected_probe_df,
        OUT_DIR / "report.pdf",
    )
    print("embedding_only")
    print(json.dumps(metrics_json, indent=2))
    print("embedding_plus_agb")
    print(json.dumps(metrics_demo_json, indent=2))
    print(f"wrote report: {OUT_DIR / 'report.pdf'}")


if __name__ == "__main__":
    main()
