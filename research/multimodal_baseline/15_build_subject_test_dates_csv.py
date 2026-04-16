"""
Build a subject-level overlap table of test dates across signal folders and tabular loaders.

Output columns include:
- registration_code
- date_<modality> (semicolon-separated YYYY-MM-DD values)
- has_<modality> (0/1)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")

from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.MentalLoader import MentalLoader
from LabData.DataLoaders.NightingaleLoader import NightingaleLoader
from LabData.DataLoaders.OlinkLoader import OlinkLoader
from LabData.DataLoaders.UntargetedMetabolomicsLoader import (
    UntargetedMetabolomicsLoader,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


OUT_CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
OUT_SUMMARY = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap_summary.txt"
OUT_PREFIXES = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_path_prefixes.csv"

STUDY_IDS = [10]

SIGNAL_SOURCES = {
    "cgm": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm",
    "sleep": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/zzp",
    "dexa": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa/dicom",
    "ecg": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/ecg/dicom",
    "retina": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes",
    "ultrasound": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/ultrasound/jpg",
    "voice": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/voice",
    "abi": "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/abi/data",
    "gait": "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/skeleton_full_data_sept/all_together",
}


def normalize_registration_code(raw_id: str) -> str:
    return f"10K_{raw_id}"


def normalize_date(raw_date: str) -> str | None:
    candidates = ["%Y%m%d", "%Y-%m-%d", "%Y_%m_%d"]
    for fmt in candidates:
        try:
            return datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def extract_id_and_date(text: str) -> tuple[str | None, str | None]:
    id_match = re.search(r"(?<!\d)(\d{10})(?!\d)", text)
    subj = normalize_registration_code(id_match.group(1)) if id_match else None

    date_match = re.search(r"((?:19|20)\d{6}|(?:19|20)\d{2}[-_]\d{2}[-_]\d{2})", text)
    date_val = normalize_date(date_match.group(1)) if date_match else None
    return subj, date_val


def progress_iter(iterable, desc: str, unit: str = "item"):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


def log(msg: str) -> None:
    print(f"[overlap] {msg}", flush=True)


def collect_dates_from_path(
    path: str,
    modality: str,
    allowed_subjects: set[str] | None = None,
    stop_early_when_allowed_seen: bool = False,
    max_subjects: int | None = None,
    max_path_hints: int = 3,
    max_files_per_subject_scan: int | None = None,
) -> tuple[dict[str, set[str]], set[str], dict[str, set[str]]]:
    by_subject_dates = defaultdict(set)
    by_subject_seen = set()
    by_subject_paths = defaultdict(set)

    if not os.path.exists(path):
        log(f"signal:{modality} path missing: {path}")
        return by_subject_dates, by_subject_seen, by_subject_paths

    log(f"signal:{modality} scanning path={path}")

    target_raw_ids = None
    if allowed_subjects is not None:
        target_raw_ids = {s.replace("10K_", "") for s in allowed_subjects}

    walk_iter = os.walk(path)
    walk_iter = progress_iter(walk_iter, f"signal:{modality}", unit="dir")

    visited_dirs = 0
    scanned_files = 0
    files_scanned_per_subject = defaultdict(int)

    for root, dirs, files in walk_iter:
        visited_dirs += 1
        if target_raw_ids is not None and root == path:
            id_dirs = [d for d in dirs if re.search(r"(?<!\d)(\d{10})(?!\d)", d)]
            filtered_dirs = []
            if id_dirs:
                for d in dirs:
                    m = re.search(r"(?<!\d)(\d{10})(?!\d)", d)
                    if m is not None and m.group(1) in target_raw_ids:
                        filtered_dirs.append(d)
                log(
                    f"signal:{modality} root filter kept {len(filtered_dirs)}/{len(dirs)} dirs for target subjects"
                )
                dirs[:] = filtered_dirs

        parent_name = os.path.basename(root)
        parent_subj, parent_date = extract_id_and_date(parent_name)
        path_subj, _ = extract_id_and_date(root)

        if (
            max_files_per_subject_scan is not None
            and path_subj is not None
            and files_scanned_per_subject[path_subj] >= max_files_per_subject_scan
        ):
            dirs[:] = []
            continue

        for fname in files:
            scanned_files += 1
            file_subj, file_date = extract_id_and_date(fname)
            subj = file_subj or parent_subj or path_subj
            date_val = file_date or parent_date

            if subj is None:
                continue
            if allowed_subjects is not None and subj not in allowed_subjects:
                continue

            if (
                max_files_per_subject_scan is not None
                and files_scanned_per_subject[subj] >= max_files_per_subject_scan
            ):
                continue
            files_scanned_per_subject[subj] += 1

            by_subject_seen.add(subj)
            if date_val is not None:
                by_subject_dates[subj].add(date_val)
            if len(by_subject_paths[subj]) < max_path_hints:
                rel_hint = os.path.relpath(os.path.join(root, fname), path)
                by_subject_paths[subj].add(rel_hint)

            if (
                max_subjects is not None
                and allowed_subjects is None
                and len(by_subject_seen) >= max_subjects
            ):
                log(
                    f"signal:{modality} early stop at max_subjects={max_subjects} "
                    f"(dirs={visited_dirs}, files={scanned_files})"
                )
                return by_subject_dates, by_subject_seen, by_subject_paths

            if (
                stop_early_when_allowed_seen
                and allowed_subjects is not None
                and by_subject_seen >= allowed_subjects
            ):
                log(
                    f"signal:{modality} early stop after covering allowed subjects "
                    f"(dirs={visited_dirs}, files={scanned_files})"
                )
                return by_subject_dates, by_subject_seen, by_subject_paths

    log(
        f"signal:{modality} done dirs={visited_dirs} files={scanned_files} "
        f"subjects={len(by_subject_seen)}"
    )
    return by_subject_dates, by_subject_seen, by_subject_paths


def pick_columns(meta: pd.DataFrame) -> tuple[str, str]:
    reg_candidates = ["RegistrationCode", "registration_code", "participant_id"]
    date_candidates = ["Date", "date", "sample_date", "SampleDate"]

    reg_col = next((c for c in reg_candidates if c in meta.columns), None)
    date_col = next((c for c in date_candidates if c in meta.columns), None)

    if reg_col is None:
        for c in meta.columns:
            if "registration" in c.lower() and "code" in c.lower():
                reg_col = c
                break

    if date_col is None:
        for c in meta.columns:
            if "date" in c.lower():
                date_col = c
                break

    if reg_col is None or date_col is None:
        raise ValueError(
            f"Could not find RegistrationCode/Date columns. Columns: {meta.columns.tolist()}"
        )

    return reg_col, date_col


def unwrap_metadata_table(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for k in ["df_metadata", "metadata", "meta"]:
            v = obj.get(k)
            if isinstance(v, pd.DataFrame):
                return v
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
            if isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, pd.DataFrame):
                        return vv
    raise ValueError(
        f"Could not unwrap metadata DataFrame from object type {type(obj)}"
    )


def normalize_subject_value(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    m = re.search(r"(?<!\d)(\d{10})(?!\d)", s)
    if m:
        return normalize_registration_code(m.group(1))
    if s.startswith("10K_"):
        return s
    return None


def normalize_date_value(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()

    direct = normalize_date(s)
    if direct:
        return direct

    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return None


def collect_dates_from_loader(
    loader_name: str,
    loader_obj,
    allowed_subjects: set[str] | None = None,
    **loader_kwargs,
) -> tuple[dict[str, set[str]], set[str]]:
    by_subject_dates = defaultdict(set)
    by_subject_seen = set()

    log(f"tabular:{loader_name} loading via {loader_obj.__class__.__name__}")
    t0 = time.time()
    if loader_name == "metabolites":
        import LabUtils.Caching as _caching

        prev_ignore_cache = _caching.IGNORE_CACHE
        _caching.IGNORE_CACHE = True
        try:
            data = loader_obj.get_data(study_ids=STUDY_IDS, **loader_kwargs)
        finally:
            _caching.IGNORE_CACHE = prev_ignore_cache
    else:
        data = loader_obj.get_data(study_ids=STUDY_IDS, **loader_kwargs)
    if hasattr(data, "df_metadata"):
        raw_meta = data.df_metadata
    else:
        raw_meta = data
    meta = unwrap_metadata_table(raw_meta).reset_index()
    log(
        f"tabular:{loader_name} loader returned rows={len(meta)} in {time.time() - t0:.1f}s"
    )
    reg_col, date_col = pick_columns(meta)

    iterator = meta[[reg_col, date_col]].iterrows()
    iterator = progress_iter(iterator, f"tabular:{loader_name}")
    kept_rows = 0
    for _, row in iterator:
        subj = normalize_subject_value(row[reg_col])
        if subj is None:
            continue
        if allowed_subjects is not None and subj not in allowed_subjects:
            continue
        by_subject_seen.add(subj)
        kept_rows += 1
        d = normalize_date_value(row[date_col])
        if d is not None:
            by_subject_dates[subj].add(d)

    log(
        f"tabular:{loader_name} done kept_rows={kept_rows} subjects={len(by_subject_seen)}"
    )

    return by_subject_dates, by_subject_seen


def format_date_set(values: set[str]) -> str:
    if not values:
        return ""
    return ";".join(sorted(values))


def format_path_set(values: set[str], limit: int) -> str:
    if not values:
        return ""
    return ";".join(sorted(values)[:limit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build overlap CSV for subject test dates."
    )
    parser.add_argument("--out-csv", default=OUT_CSV, help="Output CSV path")
    parser.add_argument(
        "--out-summary", default=OUT_SUMMARY, help="Output summary path"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Limit to first N CGM subjects for smoke testing",
    )
    parser.add_argument(
        "--skip-tabular",
        action="store_true",
        help="Skip LabData tabular loaders for fast smoke tests",
    )
    parser.add_argument(
        "--out-prefixes",
        default=OUT_PREFIXES,
        help="Output CSV mapping modality to root path",
    )
    parser.add_argument(
        "--max-path-hints",
        type=int,
        default=3,
        help="Maximum number of path hints per subject per modality",
    )
    parser.add_argument(
        "--max-files-per-subject-scan",
        type=int,
        default=None,
        help="Cap scanned files per subject per signal modality (useful for smoke tests)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    all_subjects = set()
    per_modality_dates = {}
    per_modality_seen = {}
    per_modality_paths = {}
    summary_lines = []
    summary_lines.append(f"max_subjects={args.max_subjects}")
    summary_lines.append(f"skip_tabular={args.skip_tabular}")

    anchor_subjects = None
    signal_modalities = [
        "cgm",
        "sleep",
        "dexa",
        "ecg",
        "retina",
        "ultrasound",
        "voice",
        "abi",
        "gait",
    ]

    log("starting overlap build")
    for modality in signal_modalities:
        t0 = time.time()
        src = SIGNAL_SOURCES[modality]
        dates, seen, paths = collect_dates_from_path(
            path=src,
            modality=modality,
            allowed_subjects=anchor_subjects,
            stop_early_when_allowed_seen=bool(anchor_subjects),
            max_subjects=args.max_subjects if modality == "cgm" else None,
            max_path_hints=args.max_path_hints,
            max_files_per_subject_scan=args.max_files_per_subject_scan,
        )

        if modality == "cgm" and args.max_subjects is not None:
            anchor_subjects = set(sorted(seen)[: args.max_subjects])
            seen = anchor_subjects
            dates = defaultdict(
                set, {k: v for k, v in dates.items() if k in anchor_subjects}
            )
            paths = defaultdict(
                set, {k: v for k, v in paths.items() if k in anchor_subjects}
            )

        per_modality_dates[modality] = dates
        per_modality_seen[modality] = seen
        per_modality_paths[modality] = paths
        all_subjects |= seen
        n_with_date = sum(1 for s in seen if len(dates.get(s, set())) > 0)
        summary_lines.append(
            f"signal:{modality} path={src} subjects={len(seen)} subjects_with_date={n_with_date}"
        )
        log(
            f"signal:{modality} summary subjects={len(seen)} subjects_with_date={n_with_date} "
            f"elapsed={time.time() - t0:.1f}s"
        )

    tabular_modalities = [
        "metabolites",
        "blood_test",
        "microbiome",
        "nightingale",
        "mental",
        "proteomics",
    ]
    if not args.skip_tabular:
        tabular_specs = [
            ("metabolites", UntargetedMetabolomicsLoader(gen_cache=True)),
            ("blood_test", BloodTestsLoader()),
            ("microbiome", GutMBLoader()),
            ("nightingale", NightingaleLoader()),
            ("mental", MentalLoader()),
            ("proteomics", OlinkLoader()),
        ]

        loader_kwargs = {
            "metabolites": {},
            "blood_test": {},
            "microbiome": {"df": "segal_species", "take_log": True},
            "nightingale": {},
            "mental": {},
            "proteomics": {},
        }

        for modality, loader in tabular_specs:
            t0 = time.time()
            try:
                dates, seen = collect_dates_from_loader(
                    modality,
                    loader,
                    allowed_subjects=anchor_subjects,
                    **loader_kwargs.get(modality, {}),
                )
                per_modality_dates[modality] = dates
                per_modality_seen[modality] = seen
                per_modality_paths[modality] = defaultdict(set)
                all_subjects |= seen
                n_with_date = sum(1 for s in seen if len(dates.get(s, set())) > 0)
                summary_lines.append(
                    f"tabular:{modality} loader={loader.__class__.__name__} subjects={len(seen)} subjects_with_date={n_with_date}"
                )
                log(
                    f"tabular:{modality} summary subjects={len(seen)} subjects_with_date={n_with_date} "
                    f"elapsed={time.time() - t0:.1f}s"
                )
            except Exception as exc:
                per_modality_dates[modality] = defaultdict(set)
                per_modality_seen[modality] = set()
                per_modality_paths[modality] = defaultdict(set)
                summary_lines.append(
                    f"tabular:{modality} loader={loader.__class__.__name__} ERROR={exc}"
                )
                log(f"tabular:{modality} ERROR after {time.time() - t0:.1f}s: {exc}")
    else:
        for modality in tabular_modalities:
            per_modality_dates[modality] = defaultdict(set)
            per_modality_seen[modality] = set()
            per_modality_paths[modality] = defaultdict(set)
            summary_lines.append(f"tabular:{modality} skipped")

    modalities = [
        "cgm",
        "sleep",
        "dexa",
        "ecg",
        "retina",
        "ultrasound",
        "voice",
        "abi",
        "gait",
        "metabolites",
        "blood_test",
        "microbiome",
        "nightingale",
        "mental",
        "proteomics",
    ]

    rows = []
    for subj in sorted(all_subjects):
        row = {"registration_code": subj}
        for m in modalities:
            subj_dates = per_modality_dates[m].get(subj, set())
            row[f"date_{m}"] = format_date_set(subj_dates)
            row[f"n_dates_{m}"] = len(subj_dates)
            row[f"path_hint_{m}"] = format_path_set(
                per_modality_paths[m].get(subj, set()), args.max_path_hints
            )
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    log(f"wrote csv rows={len(out_df)} path={args.out_csv}")

    presence_cols = [f"n_dates_{m}" for m in modalities]
    out_df["n_modalities"] = (out_df[presence_cols] > 0).sum(axis=1)
    summary_lines.append(f"total_subjects={len(out_df)}")
    summary_lines.append(
        "subjects_with_3plus_modalities=" + str((out_df["n_modalities"] >= 3).sum())
    )
    summary_lines.append(
        "subjects_with_5plus_modalities=" + str((out_df["n_modalities"] >= 5).sum())
    )
    summary_lines.append(
        "subjects_with_all_modalities="
        + str((out_df["n_modalities"] == len(modalities)).sum())
    )

    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    log(f"wrote summary path={args.out_summary}")

    prefix_rows = []
    for modality, src in SIGNAL_SOURCES.items():
        prefix_rows.append({"modality": modality, "root_path": src, "kind": "signal"})
    for modality in tabular_modalities:
        prefix_rows.append(
            {"modality": modality, "root_path": "loader_based", "kind": "tabular"}
        )
    pd.DataFrame(prefix_rows).to_csv(args.out_prefixes, index=False)
    log(f"wrote path prefixes csv path={args.out_prefixes}")

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved summary: {args.out_summary}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
