"""
Find the correct column names for outcomes we want to test
"""
import sys
sys.path.append('/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_body_system_df
from body_system_loader.biomarker_browser import BiomarkerBrowser

print("=" * 80)
print("FINDING OUTCOME COLUMNS")
print("=" * 80)

# Initialize browser
browser = BiomarkerBrowser()

# Shared features
print("\n[SHARED FEATURES]")
print("\nAge/Gender/BMI system:")
agb = load_body_system_df('Age_Gender_BMI')
print(f"Columns: {list(agb.columns)}")

# DEXA-specific (body composition)
print("\n[DEXA-SPECIFIC FEATURES]")
print("\nBody composition system:")
bc = load_body_system_df('body_composition')
print(f"Total columns: {len(bc.columns)}")

# Search for relevant terms
print("\nSearching for bone-related features:")
bone_cols = browser.search_biomarkers('bone', limit=20)
for col in bone_cols:
    print(f"   {col}")

print("\nSearching for fat-related features:")
fat_cols = browser.search_biomarkers('fat', limit=20)
for col in fat_cols:
    print(f"   {col}")

print("\nSearching for lean mass features:")
lean_cols = browser.search_biomarkers('lean', limit=20)
for col in lean_cols:
    print(f"   {col}")

print("\nSearching for visceral features:")
visc_cols = browser.search_biomarkers('visceral', limit=10)
for col in visc_cols:
    print(f"   {col}")

# Cross-modal features
print("\n[CROSS-MODAL FEATURES]")

print("\nGlycemic status system:")
glyc = load_body_system_df('glycemic_status')
print(f"Columns ({len(glyc.columns)}): {list(glyc.columns)}")

print("\nBlood lipids system:")
lipids = load_body_system_df('blood_lipids')
print(f"Total columns: {len(lipids.columns)}")
print("\nSearching for cholesterol features:")
chol_cols = browser.search_biomarkers('cholesterol', limit=10)
for col in chol_cols:
    print(f"   {col}")

print("\nSearching for triglyceride features:")
trig_cols = browser.search_biomarkers('triglyceride', limit=10)
for col in trig_cols:
    print(f"   {col}")

# Retina-specific - need to think about what could be retina-specific
print("\n[RETINA-SPECIFIC FEATURES - CANDIDATES]")
print("\nCardiovascular features (vascular health visible in retina):")
cardio = load_body_system_df('cardiovascular_system')
print(f"Total columns: {len(cardio.columns)}")
print("\nSearching for blood pressure features:")
bp_cols = browser.search_biomarkers('blood_pressure', limit=10)
for col in bp_cols:
    print(f"   {col}")

print("\nSearching for vascular features:")
vasc_cols = browser.search_biomarkers('vascular', limit=10)
for col in vasc_cols:
    print(f"   {col}")

print("\nImmune system (inflammatory markers):")
immune = load_body_system_df('immune_system')
print(f"Total columns: {len(immune.columns)}")

print("\n=" * 80)
