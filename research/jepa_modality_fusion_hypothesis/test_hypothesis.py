"""
Test the hypothesis: JEPA modality fusion maximizes shared features and suppresses modality-specific features

Strategy:
1. Select outcomes across the shared → specific spectrum:
   - SHARED: Age, BMI (should be best in pooled)
   - DEXA-SPECIFIC: Bone mineral density, visceral adipose tissue, lean mass
   - RETINA-SPECIFIC: Vascular health markers, glucose/diabetic markers
   - CROSS-MODAL: Outcomes that could plausibly be predicted from both

2. For each outcome, predict from:
   - DEXA embeddings only
   - Retina embeddings only
   - Pooled embeddings (subjects with both)

3. Compare R² performance with proper cross-validation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add LabTools to path
sys.path.append('/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_columns_as_df, load_body_system_df
from predict_and_eval.utils.ids_folds import ids_folds
from predict_and_eval.regression_seeding.Regressions import Regressions

# Paths
EMBED_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh")
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")

print("=" * 80)
print("JEPA MODALITY FUSION HYPOTHESIS TESTING")
print("=" * 80)

# ============================================================================
# 1. LOAD EMBEDDINGS
# ============================================================================
print("\n[1/6] Loading embeddings...")
dexa_emb = pd.read_csv(EMBED_DIR / "dexa_embeddings.csv")
retina_emb = pd.read_csv(EMBED_DIR / "retina_embeddings.csv")
pooled_emb = pd.read_csv(EMBED_DIR / "pooled_embeddings.csv")

# Set proper index
dexa_emb = dexa_emb.set_index(['RegistrationCode', 'research_stage'])
retina_emb = retina_emb.set_index(['RegistrationCode', 'research_stage'])
pooled_emb = pooled_emb.set_index(['RegistrationCode', 'research_stage'])

# Handle duplicates in pooled (keep first occurrence)
pooled_emb = pooled_emb[~pooled_emb.index.duplicated(keep='first')]

print(f"   DEXA: {dexa_emb.shape}")
print(f"   Retina: {retina_emb.shape}")
print(f"   Pooled (deduplicated): {pooled_emb.shape}")

# ============================================================================
# 2. SELECT OUTCOMES TO TEST
# ============================================================================
print("\n[2/6] Selecting outcomes across shared-specific spectrum...")

# Define outcome categories
outcomes_to_test = {
    # SHARED features (age, BMI) - should be BEST in pooled
    'shared': {
        'Age': 'age',
        'BMI': 'bmi',
    },

    # DEXA-SPECIFIC features - should be BEST in DEXA-only
    'dexa_specific': {
        'Bone Mineral Density': 'total_bone_mineral_density',
        'Visceral Adipose Tissue': 'visceral_adipose_tissue_mass',
        'Total Lean Mass': 'total_lean_mass',
        'Body Fat Percentage': 'body_fat_percentage',
    },

    # RETINA-SPECIFIC features - need to identify good candidates
    'retina_specific': {
        # We'll discover these from the data
    },

    # CROSS-MODAL features that could plausibly come from either
    'cross_modal': {
        'HbA1c': 'hba1c',
        'Glucose': 'glucose_fasting',
        'Triglycerides': 'triglycerides',
        'HDL Cholesterol': 'hdl_cholesterol',
        'LDL Cholesterol': 'ldl_cholesterol',
        'C-Reactive Protein': 'c_reactive_protein',
    }
}

# Flatten for loading
all_outcomes = {}
for category, outcomes in outcomes_to_test.items():
    all_outcomes.update(outcomes)

print(f"   Testing {len(all_outcomes)} outcomes across {len(outcomes_to_test)} categories")

# ============================================================================
# 3. LOAD OUTCOME DATA
# ============================================================================
print("\n[3/6] Loading outcome data...")

# Load Age, Gender, BMI
age_gender_bmi = load_body_system_df('Age_Gender_BMI')
print(f"   Age/Gender/BMI: {age_gender_bmi.shape}")

# Load body composition (DEXA-specific)
body_comp = load_body_system_df('body_composition')
print(f"   Body composition: {body_comp.shape}")

# Load blood lipids (cross-modal)
blood_lipids = load_body_system_df('blood_lipids')
print(f"   Blood lipids: {blood_lipids.shape}")

# Load glycemic status (cross-modal)
glycemic = load_body_system_df('glycemic_status')
print(f"   Glycemic status: {glycemic.shape}")

# ============================================================================
# 4. PREPARE DATASETS FOR EACH EMBEDDING TYPE
# ============================================================================
print("\n[4/6] Preparing datasets...")

def prepare_dataset(embeddings, outcomes_df, outcome_col):
    """Merge embeddings with outcome, drop NaNs"""
    # Ensure outcomes_df has the column
    if outcome_col not in outcomes_df.columns:
        print(f"   WARNING: {outcome_col} not found in outcome data")
        return None

    # Merge on index
    df = embeddings.join(outcomes_df[[outcome_col]], how='inner')

    # Drop NaNs in outcome
    df = df.dropna(subset=[outcome_col])

    if len(df) == 0:
        print(f"   WARNING: No samples after merge for {outcome_col}")
        return None

    # Split X and y
    X = df.drop(columns=[outcome_col])
    y = df[outcome_col]

    return X, y

# ============================================================================
# 5. RUN CROSS-VALIDATION FOR EACH OUTCOME × EMBEDDING TYPE
# ============================================================================
print("\n[5/6] Running cross-validation...")

results = []

for category, outcomes in outcomes_to_test.items():
    print(f"\n   Category: {category.upper()}")

    for outcome_name, outcome_col in outcomes.items():
        print(f"\n   Testing: {outcome_name} ({outcome_col})")

        # Determine which DataFrame has this outcome
        outcome_df = None
        if outcome_col in age_gender_bmi.columns:
            outcome_df = age_gender_bmi
        elif outcome_col in body_comp.columns:
            outcome_df = body_comp
        elif outcome_col in blood_lipids.columns:
            outcome_df = blood_lipids
        elif outcome_col in glycemic.columns:
            outcome_df = glycemic

        if outcome_df is None:
            print(f"      ⚠ Outcome not found, skipping")
            continue

        # Test on each embedding type
        for emb_name, embeddings in [('DEXA', dexa_emb), ('Retina', retina_emb), ('Pooled', pooled_emb)]:
            try:
                # Prepare data
                data = prepare_dataset(embeddings, outcome_df, outcome_col)
                if data is None:
                    continue

                X, y = data

                print(f"      {emb_name}: n={len(X)}", end=" ")

                # Create CV folds (subject-level split)
                cv_folds = ids_folds(
                    pd.DataFrame({'y': y}),
                    seeds=[42],  # Single seed for now
                    n_splits=5
                )[0]

                # Cross-validate
                regressions = Regressions()
                result = regressions.cross_validate_model(
                    X, y, cv_folds,
                    model_key='LR_ridge'  # Linear model for embeddings
                )

                # Evaluate
                eval_result = regressions.evaluate_predictions(X, y, result['predictions'])

                r2 = eval_result['r2']
                print(f"R² = {r2:.3f}")

                # Store result
                results.append({
                    'category': category,
                    'outcome': outcome_name,
                    'outcome_col': outcome_col,
                    'embedding_type': emb_name,
                    'n_samples': len(X),
                    'r2': r2,
                    'spearman': eval_result.get('spearman', np.nan),
                    'mae': eval_result.get('mae', np.nan),
                })

            except Exception as e:
                print(f"      {emb_name}: ERROR - {str(e)}")
                continue

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)

print(f"   Saved to: {OUTPUT_DIR / 'hypothesis_test_results.csv'}")

# Print summary
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

for category in outcomes_to_test.keys():
    cat_results = results_df[results_df['category'] == category]
    if len(cat_results) == 0:
        continue

    print(f"\n{category.upper()}:")
    pivot = cat_results.pivot_table(
        index='outcome',
        columns='embedding_type',
        values='r2',
        aggfunc='mean'
    )
    print(pivot.to_string())

    # Highlight best performing embedding for each outcome
    print("\nBest embedding per outcome:")
    for outcome in pivot.index:
        best_emb = pivot.loc[outcome].idxmax()
        best_r2 = pivot.loc[outcome].max()
        print(f"   {outcome}: {best_emb} (R²={best_r2:.3f})")

print("\n" + "=" * 80)
print("HYPOTHESIS TESTING COMPLETE")
print("=" * 80)
