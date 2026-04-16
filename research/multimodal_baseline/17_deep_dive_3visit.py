"""
Deep dive: find the best modality combinations for maximizing 3-visit subjects.
Also explore 2+ visit as a more lenient threshold.
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
from itertools import combinations

CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
df = pd.read_csv(CSV)

modalities = []
for col in df.columns:
    if col.startswith("n_dates_"):
        modalities.append(col.replace("n_dates_", ""))

print("=" * 80)
print("PART 1: BEST 4-MODALITY COMBOS FOR 3+ VISITS")
print("=" * 80)

# Focus on modalities that have meaningful 3+ visit counts
viable_mods = [m for m in modalities if (df[f"n_dates_{m}"] >= 3).sum() >= 100]
print(f"Viable modalities (100+ subjects at 3+ visits): {viable_mods}")
print()

best_4 = []
for combo in combinations(viable_mods, 4):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    count = mask.sum()
    if count > 0:
        best_4.append((combo, count))

best_4.sort(key=lambda x: x[1], reverse=True)
print("Top 15 four-modality combos (3+ visits in ALL):")
for combo, count in best_4[:15]:
    print(f"  {' + '.join(combo):65s}: {count:5d}")
print()

print("=" * 80)
print("PART 2: BEST 5-MODALITY COMBOS FOR 3+ VISITS")
print("=" * 80)

best_5 = []
for combo in combinations(viable_mods, 5):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    count = mask.sum()
    if count > 0:
        best_5.append((combo, count))

best_5.sort(key=lambda x: x[1], reverse=True)
print("Top 10 five-modality combos (3+ visits in ALL):")
for combo, count in best_5[:10]:
    print(f"  {' + '.join(combo):75s}: {count:5d}")
print()

print("=" * 80)
print("PART 3: BEST 3-MODALITY COMBOS FOR 2+ VISITS (lenient)")
print("=" * 80)

best_3_2plus = []
for combo in combinations(viable_mods, 3):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 2
    count = mask.sum()
    best_3_2plus.append((combo, count))

best_3_2plus.sort(key=lambda x: x[1], reverse=True)
print("Top 10 three-modality combos (2+ visits in ALL):")
for combo, count in best_3_2plus[:10]:
    print(f"  {' + '.join(combo):65s}: {count:5d}")
print()

print("=" * 80)
print("PART 4: BEST 4-MODALITY COMBOS FOR 2+ VISITS")
print("=" * 80)

best_4_2plus = []
for combo in combinations(viable_mods, 4):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 2
    count = mask.sum()
    if count > 0:
        best_4_2plus.append((combo, count))

best_4_2plus.sort(key=lambda x: x[1], reverse=True)
print("Top 10 four-modality combos (2+ visits in ALL):")
for combo, count in best_4_2plus[:10]:
    print(f"  {' + '.join(combo):75s}: {count:5d}")
print()

print("=" * 80)
print("PART 5: DEEP DIVE ON BEST 3-VISIT TRIPLE")
print("=" * 80)

best_triple = best_4[0][0][:3] if best_4 else None
if best_triple:
    # Actually find best triple
    best_triple = None
    best_triple_count = 0
    for combo in combinations(viable_mods, 3):
        mask = pd.Series(True, index=df.index)
        for mod in combo:
            mask &= df[f"n_dates_{mod}"] >= 3
        count = mask.sum()
        if count > best_triple_count:
            best_triple_count = count
            best_triple = combo

    print(f"Best triple: {best_triple} → {best_triple_count} subjects")

    subset = df.copy()
    for mod in best_triple:
        subset = subset[subset[f"n_dates_{mod}"] >= 3]

    print(
        f"\nFor these {len(subset)} subjects, additional modality coverage at 3+ visits:"
    )
    for mod in modalities:
        if mod not in best_triple:
            n = (subset[f"n_dates_{mod}"] >= 3).sum()
            n2 = (subset[f"n_dates_{mod}"] >= 2).sum()
            n1 = (subset[f"n_dates_{mod}"] >= 1).sum()
            print(f"  {mod:20s}: 1+={n1:5d}, 2+={n2:5d}, 3+={n:5d}")
print()

print("=" * 80)
print("PART 6: CGM-SPECIFIC ANALYSIS (why is CGM so low at 3+ visits?)")
print("=" * 80)

cgm_dates = df["n_dates_cgm"].value_counts().sort_index()
print("CGM visit distribution:")
for n, count in cgm_dates.items():
    print(f"  {n} visits: {count}")

# What if we use CGM at 1+ and other modalities at 3+?
print("\nCGM at 1+ combined with top triples at 3+:")
top_triples_3plus = [(c, cnt) for c, cnt in best_4[:20]]
for combo, _ in top_triples_3plus[:5]:
    mask = df["n_dates_cgm"] >= 1
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    print(f"  CGM(1+) + {' + '.join(combo)}(3+): {mask.sum():5d}")

print()
print("=" * 80)
print("PART 7: SLEEP AS CGM REPLACEMENT")
print("=" * 80)

# Sleep has much better 3+ visit coverage than CGM
# Check: sleep(3+) + blood_test(3+) + microbiome(3+) + retina(3+)
combos_with_sleep = [
    ("sleep", "blood_test", "microbiome", "retina"),
    ("sleep", "blood_test", "microbiome", "mental"),
    ("sleep", "blood_test", "retina", "abi"),
    ("sleep", "microbiome", "retina", "abi"),
    ("sleep", "blood_test", "nightingale", "metabolites"),
]

for combo in combos_with_sleep:
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= 3
    print(f"  {' + '.join(combo)}(3+): {mask.sum():5d}")

print()
print("=" * 80)
print("PART 8: RECOMMENDED COMBINATIONS (balancing 3-visit count + modality diversity)")
print("=" * 80)

# Show a range of options
recommendations = [
    ("Max 3-visit count (3 mods)", ("blood_test", "mental", "microbiome"), 3),
    ("Max 3-visit count (4 mods)", best_4[0][0] if best_4 else None, 3),
    ("Rich signal + tabular (3 mods)", ("sleep", "retina", "blood_test"), 3),
    (
        "Rich signal + tabular (4 mods)",
        ("sleep", "retina", "blood_test", "microbiome"),
        3,
    ),
    ("Include metabolites (3 mods)", ("blood_test", "microbiome", "metabolites"), 3),
    ("Include nightingale (3 mods)", ("blood_test", "microbiome", "nightingale"), 3),
    ("Include proteomics (2+ visits)", ("blood_test", "microbiome", "proteomics"), 2),
    ("Include DEXA (2+ visits)", ("blood_test", "retina", "dexa"), 2),
]

for name, combo, threshold in recommendations:
    if combo is None:
        print(f"  {name}: N/A")
        continue
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= threshold
    print(f"  {name}: {combo} ({threshold}+ visits) → {mask.sum():5d} subjects")
