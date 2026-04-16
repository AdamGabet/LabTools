"""
Analyze modality-visit coverage to maximize 3-visit subject count.
Focus: find modality combinations that give the most subjects with 3+ visits.
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import numpy as np

CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
df = pd.read_csv(CSV)

print(f"Total subjects: {len(df)}")
print()

# ── 1. Per-modality: how many subjects have 1, 2, 3+ dates? ──
print("=" * 70)
print("PER-MODALITY VISIT COUNTS")
print("=" * 70)

modalities = []
for col in df.columns:
    if col.startswith("n_dates_"):
        modalities.append(col.replace("n_dates_", ""))

visit_stats = []
for mod in modalities:
    col = f"n_dates_{mod}"
    n0 = (df[col] == 0).sum()
    n1 = (df[col] == 1).sum()
    n2 = (df[col] == 2).sum()
    n3plus = (df[col] >= 3).sum()
    max_dates = df[col].max()
    visit_stats.append(
        {
            "modality": mod,
            "no_data": n0,
            "1_visit": n1,
            "2_visits": n2,
            "3plus_visits": n3plus,
            "max_dates": max_dates,
        }
    )

stats_df = pd.DataFrame(visit_stats).sort_values("3plus_visits", ascending=False)
print(stats_df.to_string(index=False))
print()

# ── 2. Current 4 modalities: how many subjects have 3+ visits in ALL? ──
current_mods = ["cgm", "dexa", "retina", "metabolites"]
print("=" * 70)
print(f"CURRENT MODALITIES: {current_mods}")
print("=" * 70)

for n_visits in [1, 2, 3, 4]:
    mask = pd.Series(True, index=df.index)
    for mod in current_mods:
        mask &= df[f"n_dates_{mod}"] >= n_visits
    print(
        f"Subjects with {n_visits}+ visits in ALL {len(current_mods)} modalities: {mask.sum()}"
    )

print()

# ── 3. Per-modality: how many subjects have 3+ visits? ──
print("=" * 70)
print("MODALITIES RANKED BY 3+ VISIT SUBJECTS")
print("=" * 70)
for _, row in stats_df.iterrows():
    print(
        f"  {row['modality']:20s} | 3+ visits: {row['3plus_visits']:5d} | max dates: {row['max_dates']}"
    )
print()

# ── 4. Intersection analysis: which modality combos give most 3-visit subjects? ──
print("=" * 70)
print("TOP MODALITIES FOR 3+ VISITS (individual)")
print("=" * 70)
three_visit_mods = stats_df[stats_df["3plus_visits"] > 0].sort_values(
    "3plus_visits", ascending=False
)
for _, row in three_visit_mods.head(10).iterrows():
    mod = row["modality"]
    n3 = row["3plus_visits"]
    print(f"  {mod:20s}: {n3} subjects with 3+ visits")
print()

# ── 5. Intersection of top modalities for 3+ visits ──
print("=" * 70)
print("INTERSECTIONS: subjects with 3+ visits in multiple modalities")
print("=" * 70)

# Get top modalities by 3+ visit count
top_mods = three_visit_mods.head(8)["modality"].tolist()
print(f"Top modalities: {top_mods}")
print()

# Pairwise intersections
print("Pairwise intersections (3+ visits in BOTH):")
for i, mod1 in enumerate(top_mods[:6]):
    for mod2 in top_mods[i + 1 : 6]:
        mask = (df[f"n_dates_{mod1}"] >= 3) & (df[f"n_dates_{mod2}"] >= 3)
        print(f"  {mod1:20s} + {mod2:20s}: {mask.sum():5d}")
print()

# ── 6. Triple intersections for 3+ visits ──
print("Triple intersections (3+ visits in ALL THREE):")
from itertools import combinations

for combo in combinations(top_mods[:6], 3):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    print(f"  {' + '.join(combo):55s}: {mask.sum():5d}")
print()

# ── 7. What if we relax to 2+ visits? ──
print("=" * 70)
print("INTERSECTIONS: subjects with 2+ visits (more lenient)")
print("=" * 70)

for combo in combinations(top_mods[:8], 3):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 2
    print(f"  {' + '.join(combo):55s}: {mask.sum():5d}")
print()

# ── 8. Maximum modality count per subject at 3+ visits ──
print("=" * 70)
print("MAX MODALITIES PER SUBJECT AT 3+ VISITS")
print("=" * 70)

df["n_mods_3plus"] = 0
for mod in modalities:
    df["n_mods_3plus"] += (df[f"n_dates_{mod}"] >= 3).astype(int)

for n in range(len(modalities), -1, -1):
    count = (df["n_mods_3plus"] >= n).sum()
    if count > 0:
        print(f"  Subjects with {n:2d}+ modalities at 3+ visits: {count}")
print()

# ── 9. For subjects with 3+ visits in top modalities, what other modalities do they have? ──
print("=" * 70)
print("COVERAGE OF OTHER MODALITIES for subjects with 3+ visits in TOP 3")
print("=" * 70)

# Find best triple for 3+ visits
best_triple_count = 0
best_triple = None
for combo in combinations(three_visit_mods["modality"].tolist()[:8], 3):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    if mask.sum() > best_triple_count:
        best_triple_count = mask.sum()
        best_triple = combo

print(f"Best triple for 3+ visits: {best_triple} ({best_triple_count} subjects)")
print()

# For these subjects, check coverage of all other modalities
subset = df[pd.Series(True, index=df.index)]
for mod in modalities:
    for n in [1, 2, 3]:
        mask = pd.Series(True, index=df.index)
        for m in best_triple:
            mask &= df[f"n_dates_{m}"] >= 3
        n_have = (mask & (df[f"n_dates_{mod}"] >= n)).sum()
        if n_have > 0 and mod not in best_triple:
            print(
                f"  Of these {best_triple_count} subjects, {n_have:5d} also have {mod:20s} at {n}+ visits"
            )
print()

# ── 10. Visit distribution for top modalities ──
print("=" * 70)
print("VISIT DISTRIBUTION for top 3+ visit modalities")
print("=" * 70)

for mod in three_visit_mods.head(5)["modality"].tolist():
    col = f"n_dates_{mod}"
    date_col = f"date_{mod}"
    subset = df[df[col] > 0]

    # Count unique dates across all subjects
    all_dates = set()
    for dates_str in subset[date_col].dropna():
        for d in str(dates_str).split(";"):
            d = d.strip()
            if d:
                all_dates.add(d)

    print(
        f"  {mod:20s}: {len(all_dates)} unique dates, {len(subset)} subjects, max {df[col].max()} dates/subject"
    )
