"""
List all columns from relevant body systems
"""
import sys
sys.path.append('/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_body_system_df

print("=" * 80)
print("BODY SYSTEM COLUMNS")
print("=" * 80)

# Shared
print("\n[1] Age_Gender_BMI:")
agb = load_body_system_df('Age_Gender_BMI')
for col in agb.columns:
    print(f"   {col}")

# DEXA-specific
print("\n[2] body_composition (118 columns):")
bc = load_body_system_df('body_composition')
# Group by prefix
prefixes = {}
for col in bc.columns:
    prefix = col.split('_')[0] if '_' in col else col
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(col)

for prefix in sorted(prefixes.keys()):
    print(f"\n   {prefix.upper()}:")
    for col in sorted(prefixes[prefix])[:5]:  # Show first 5
        print(f"      {col}")
    if len(prefixes[prefix]) > 5:
        print(f"      ... ({len(prefixes[prefix])} total)")

# Glycemic
print("\n[3] glycemic_status:")
glyc = load_body_system_df('glycemic_status')
for col in sorted(glyc.columns)[:20]:
    print(f"   {col}")
if len(glyc.columns) > 20:
    print(f"   ... ({len(glyc.columns)} total)")

# Blood lipids
print("\n[4] blood_lipids (top matches):")
lipids = load_body_system_df('blood_lipids')
# Look for simple names
simple_lipids = [c for c in lipids.columns if len(c.split('_')) <= 3 and 'cholesterol' in c.lower() or 'triglyceride' in c.lower()]
for col in sorted(simple_lipids)[:15]:
    print(f"   {col}")

# Cardiovascular
print("\n[5] cardiovascular_system (for retina-specific):")
cardio = load_body_system_df('cardiovascular_system')
for col in sorted(cardio.columns)[:20]:
    print(f"   {col}")
if len(cardio.columns) > 20:
    print(f"   ... ({len(cardio.columns)} total)")

# Immune (inflammation)
print("\n[6] immune_system (for inflammation markers):")
immune = load_body_system_df('immune_system')
for col in sorted(immune.columns)[:20]:
    print(f"   {col}")
if len(immune.columns) > 20:
    print(f"   ... ({len(immune.columns)} total)")

print("\n=" * 80)
