"""
Search for workable multimodal raw-data cohorts.

Goals:
- quantify fixed visit-pair overlap for raw modalities
- quantify the same overlap after the original 8-KPI filter
- test ECG as a possible replacement for CGM
- surface the first combinations that are actually usable
"""
import os
import sys
from itertools import combinations
from pathlib import Path

import h5py
import pandas as pd

sys.path.insert(0, "/home/adamgab/PycharmProjects")
sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")

from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabTools.utils.date_to_research_stage import get_date_and_research
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = Path("/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline")
OUT_TXT = OUTDIR / "viable_raw_cohorts.txt"
OUT_CSV = OUTDIR / "viable_raw_cohorts.csv"

CGM_ROOT = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm")
ECG_TEXT_ROOT = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/ecg/text")
DEXA_H5 = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5")
RETINA_H5 = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5")

VISIT_PAIRS = [
    ("baseline", "02_00_visit"),
    ("baseline", "04_00_visit"),
    ("02_00_visit", "04_00_visit"),
]
STAGES = ["baseline", "02_00_visit", "04_00_visit", "06_00_visit"]

ALL_KPIS = [
    "bt__hba1c",
    "bt__glucose",
    "total_scan_vat_mass",
    "body_total_bmd",
    "sitting_blood_pressure_systolic",
    "intima_media_th_mm_1_intima_media_thickness",
    "urate",
    "Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin",
]

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def norm_stage(stage):
    return "baseline" if stage == "00_00_visit" else stage


def load_stage_mapping():
    loader = BodyMeasuresLoader()
    body = loader.get_data()
    mapping = get_date_and_research(body).copy()
    mapping["visit_date"] = pd.to_datetime(mapping["Date"]).dt.tz_localize(None)
    mapping["research_stage"] = mapping["research_stage"].map(norm_stage)
    mapping = mapping.sort_values(["RegistrationCode", "visit_date", "research_stage"])
    mapping = mapping.drop_duplicates(["RegistrationCode", "research_stage"], keep="first")
    return mapping[["RegistrationCode", "visit_date", "research_stage"]].reset_index(drop=True)


def infer_stage_from_date(registration_code, raw_date, mapping):
    subset = mapping.loc[mapping["RegistrationCode"] == registration_code]
    if subset.empty:
        return None
    valid = subset.loc[subset["visit_date"] >= raw_date]
    if not valid.empty:
        return str(valid.iloc[0]["research_stage"])
    return str(subset.iloc[-1]["research_stage"])


def collect_cgm_idx(mapping):
    idx = set()
    for name in os.listdir(CGM_ROOT):
        path = CGM_ROOT / name

        if path.is_dir() and name.isdigit():
            reg = f"10K_{name}"
            for raw_stage in os.listdir(path):
                stage_dir = path / raw_stage
                if stage_dir.is_dir() and any(f.endswith(".txt") for f in os.listdir(stage_dir)):
                    idx.add((reg, norm_stage(raw_stage)))
            continue

        if path.is_file() and name.endswith(".txt") and "_" in name:
            sid, date_part = name.split("_", 1)
            if not sid.isdigit():
                continue
            reg = f"10K_{sid}"
            raw_date = pd.to_datetime(date_part.removesuffix(".txt"), format="%Y%m%d")
            stage = infer_stage_from_date(reg, raw_date, mapping)
            if stage is not None:
                idx.add((reg, stage))

    return idx


def collect_ecg_idx():
    idx = set()
    for subject in os.listdir(ECG_TEXT_ROOT):
        subj_dir = ECG_TEXT_ROOT / subject
        if not subject.isdigit() or not subj_dir.is_dir():
            continue
        reg = f"10K_{subject}"
        for raw_stage in os.listdir(subj_dir):
            stage_dir = subj_dir / raw_stage
            if stage_dir.is_dir() and any(f.endswith(".txt") for f in os.listdir(stage_dir)):
                idx.add((reg, norm_stage(raw_stage)))
    return idx


def collect_h5_idx(path):
    with h5py.File(path, "r") as f:
        idx = set()
        for key in f.keys():
            parts = key.split("_")
            idx.add(("_".join(parts[:2]), norm_stage("_".join(parts[2:]))))
        return idx


def collect_metabolite_idx():
    met = load_body_system_df("metabolites_annotated")
    return {(reg, norm_stage(stage)) for reg, stage in met.index}


def stage_subjects(idx, stage):
    return {reg for reg, current_stage in idx if current_stage == stage}


def kpi_subjects_by_visit():
    cgm_df = load_body_system_df("glycemic_status")
    bc_df = load_body_system_df("body_composition")
    bd_df = load_body_system_df("bone_density")
    cardio_df = load_body_system_df("cardiovascular_system")
    met_df = load_body_system_df("metabolites_annotated")

    def xs(df, visit, cols):
        if visit in df.index.get_level_values(1).unique():
            return df.xs(visit, level="research_stage")[cols]
        return pd.DataFrame(columns=cols)

    out = {}
    for visit in ["baseline", "02_00_visit", "04_00_visit"]:
        merged = (
            xs(cgm_df, visit, ["bt__hba1c", "bt__glucose"])
            .join(xs(bc_df, visit, ["total_scan_vat_mass"]), how="outer")
            .join(xs(bd_df, visit, ["body_total_bmd"]), how="outer")
            .join(
                xs(cardio_df, visit, ["sitting_blood_pressure_systolic", "intima_media_th_mm_1_intima_media_thickness"]),
                how="outer",
            )
            .join(
                xs(met_df, visit, ["urate", "Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin"]),
                how="outer",
            )
        )
        out[visit] = set(merged[merged[ALL_KPIS].notna().all(axis=1)].index)
    return out


def subject_stage_map(idx):
    out = {}
    for reg, stage in idx:
        out.setdefault(reg, set()).add(stage)
    return out


def count_fixed_pair(modality_maps, pair):
    visit_a, visit_b = pair
    subjects = []
    universe = set().union(*[set(modality_maps[name].keys()) for name in modality_maps])
    for reg in universe:
        if all(visit_a in modality_maps[name].get(reg, set()) and visit_b in modality_maps[name].get(reg, set()) for name in modality_maps):
            subjects.append(reg)
    return set(subjects)


def count_any_two(modality_maps):
    subjects = {}
    universe = set().union(*[set(modality_maps[name].keys()) for name in modality_maps])
    for reg in universe:
        shared = set(STAGES)
        for name in modality_maps:
            shared &= modality_maps[name].get(reg, set())
        if len(shared) >= 2:
            subjects[reg] = tuple(sorted(shared))
    return subjects


def main():
    log("=== Loading raw modality indices ===")
    stage_mapping = load_stage_mapping()
    raw_indices = {
        "cgm": collect_cgm_idx(stage_mapping),
        "ecg": collect_ecg_idx(),
        "dexa": collect_h5_idx(DEXA_H5),
        "retina": collect_h5_idx(RETINA_H5),
        "met": collect_metabolite_idx(),
    }
    for name, idx in raw_indices.items():
        by_stage = {stage: len(stage_subjects(idx, stage)) for stage in STAGES}
        log(f"{name}: {by_stage}")

    modality_maps = {name: subject_stage_map(idx) for name, idx in raw_indices.items()}
    kpi_map = kpi_subjects_by_visit()

    combos = [
        ("cgm", "dexa", "retina", "met"),
        ("ecg", "dexa", "retina", "met"),
        ("dexa", "retina", "met"),
        ("cgm", "retina", "met"),
        ("ecg", "retina", "met"),
        ("cgm", "dexa", "met"),
        ("ecg", "dexa", "met"),
        ("cgm", "dexa", "retina"),
        ("ecg", "dexa", "retina"),
        ("dexa", "retina"),
        ("cgm", "dexa"),
        ("cgm", "retina"),
        ("ecg", "dexa"),
        ("ecg", "retina"),
        ("cgm", "ecg"),
        ("dexa", "met"),
        ("retina", "met"),
        ("cgm", "met"),
        ("ecg", "met"),
    ]

    rows = []
    log("\n=== Fixed visit pairs ===")
    for combo in combos:
        combo_maps = {name: modality_maps[name] for name in combo}
        log(f"\ncombo={combo}")
        for pair in VISIT_PAIRS:
            raw_subjects = count_fixed_pair(combo_maps, pair)
            raw_kpi_subjects = {
                reg for reg in raw_subjects if reg in kpi_map.get(pair[0], set()) and reg in kpi_map.get(pair[1], set())
            }
            log(f"  pair={pair}: raw={len(raw_subjects)}, raw+kpi={len(raw_kpi_subjects)}")
            rows.append(
                {
                    "combo": "+".join(combo),
                    "mode": "fixed_pair",
                    "pair": f"{pair[0]}__{pair[1]}",
                    "raw_subjects": len(raw_subjects),
                    "raw_plus_kpi_subjects": len(raw_kpi_subjects),
                }
            )

    log("\n=== Any two visits ===")
    for combo in combos:
        combo_maps = {name: modality_maps[name] for name in combo}
        subjects = count_any_two(combo_maps)
        log(f"combo={combo}: subjects={len(subjects)}")
        sample = list(subjects.items())[:10]
        log(f"  sample={sample}")
        rows.append(
            {
                "combo": "+".join(combo),
                "mode": "any_two_visits",
                "pair": "any_two",
                "raw_subjects": len(subjects),
                "raw_plus_kpi_subjects": "",
            }
        )

    log("\n=== Best 3-modality fixed-pair overlaps ===")
    for combo in combinations(["cgm", "ecg", "dexa", "retina", "met"], 3):
        combo_maps = {name: modality_maps[name] for name in combo}
        best_pair = None
        best_count = -1
        for pair in VISIT_PAIRS:
            raw_subjects = count_fixed_pair(combo_maps, pair)
            if len(raw_subjects) > best_count:
                best_count = len(raw_subjects)
                best_pair = pair
        log(f"{combo}: best_pair={best_pair}, raw={best_count}")

    log("\n=== Best 2-modality fixed-pair overlaps ===")
    for combo in combinations(["cgm", "ecg", "dexa", "retina", "met"], 2):
        combo_maps = {name: modality_maps[name] for name in combo}
        best_pair = None
        best_count = -1
        for pair in VISIT_PAIRS:
            raw_subjects = count_fixed_pair(combo_maps, pair)
            if len(raw_subjects) > best_count:
                best_count = len(raw_subjects)
                best_pair = pair
        log(f"{combo}: best_pair={best_pair}, raw={best_count}")

    log("\n=== Subject-level overlap without stage constraints ===")
    modality_subjects = {name: set(current.keys()) for name, current in modality_maps.items()}
    for combo in [
        ("cgm", "dexa", "retina", "met"),
        ("ecg", "dexa", "retina", "met"),
        ("dexa", "retina", "met"),
        ("cgm", "dexa", "met"),
        ("cgm", "retina", "met"),
        ("cgm", "met"),
        ("dexa", "met"),
        ("dexa", "retina"),
        ("ecg", "dexa"),
    ]:
        subjects = set.intersection(*(modality_subjects[name] for name in combo))
        log(f"{combo}: any-visit subjects={len(subjects)}")

        multi_visit = []
        for reg in subjects:
            if all(len(modality_maps[name].get(reg, set())) >= 2 for name in combo):
                multi_visit.append(reg)
        log(f"  {combo}: subjects with >=2 visits in every modality={len(multi_visit)}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text("\n".join(lines))
    log(f"\nSaved: {OUT_TXT}")
    log(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
