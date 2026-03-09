"""
Test the hypothesis: JEPA modality fusion maximizes shared features and suppresses modality-specific features

UPDATED with correct column names and gender for confound control
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_body_system_df
from predict_and_eval.utils.ids_folds import ids_folds
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
EMBED_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh")
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")

print("=" * 80)
print("JEPA MODALITY FUSION HYPOTHESIS TESTING V2")
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
# 2. SELECT OUTCOMES TO TEST (CORRECTED COLUMN NAMES)
# ============================================================================
print("\n[2/6] Selecting outcomes...")

outcomes_to_test = {
    # SHARED features - should be BEST in pooled
    'shared': [
        ('age', 'Age_Gender_BMI', 'Age'),
        ('bmi', 'Age_Gender_BMI', 'BMI'),
    ],

    # DEXA-SPECIFIC features - should be BEST in DEXA-only
    'dexa_specific': [
        ('total_scan_vat_mass', 'body_composition', 'Visceral Adipose Tissue Mass'),
        ('total_scan_sat_mass', 'body_composition', 'Subcutaneous Adipose Tissue Mass'),
        ('body_comp_total_bone_mineral_density', 'body_composition', 'Total Bone Mineral Density'),
        ('body_comp_total_lean_mass', 'body_composition', 'Total Lean Mass'),
        ('body_comp_total_fat_percentage', 'body_composition', 'Body Fat Percentage'),
        ('weight', 'body_composition', 'Weight'),
    ],

    # RETINA-SPECIFIC features - should be BEST in Retina-only
    'retina_specific': [
        ('automorph_artery_average_width', 'cardiovascular_system', 'Retinal Artery Width'),
        ('automorph_vein_average_width', 'cardiovascular_system', 'Retinal Vein Width'),
        ('automorph_artery_tortuosity_density', 'cardiovascular_system', 'Artery Tortuosity'),
        ('automorph_cdr_vertical', 'cardiovascular_system', 'Cup-to-Disc Ratio'),
        ('automorph_fractal_dimension', 'cardiovascular_system', 'Retinal Fractal Dimension'),
    ],

    # CROSS-MODAL features - could come from either
    'cross_modal': [
        ('bt__glucose', 'glycemic_status', 'Glucose'),
        ('bt__hba1c', 'glycemic_status', 'HbA1c'),
        ('bt__wbc', 'immune_system', 'White Blood Cell Count'),
    ]
}

total_outcomes = sum(len(v) for v in outcomes_to_test.values())
print(f"   Testing {total_outcomes} outcomes across {len(outcomes_to_test)} categories")

# ============================================================================
# 3. LOAD OUTCOME DATA
# ============================================================================
print("\n[3/6] Loading outcome data...")

data_cache = {}
for category, outcomes in outcomes_to_test.items():
    for outcome_col, system, _ in outcomes:
        if system not in data_cache:
            data_cache[system] = load_body_system_df(system)
            print(f"   Loaded {system}: {data_cache[system].shape}")

# Also load gender for confound control
age_gender_bmi = data_cache['Age_Gender_BMI']

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

def prepare_dataset(embeddings, outcome_col, system):
    """Merge embeddings with outcome + gender, drop NaNs"""
    outcome_df = data_cache[system]

    if outcome_col not in outcome_df.columns:
        print(f"   ERROR: {outcome_col} not in {system}")
        return None

    # Merge outcome
    df = embeddings.join(outcome_df[[outcome_col]], how='inner')

    # Merge gender
    df = df.join(age_gender_bmi[['gender']], how='inner')

    # Drop NaNs
    df = df.dropna(subset=[outcome_col, 'gender'])

    if len(df) == 0:
        return None

    # Split X and y
    embedding_cols = [c for c in df.columns if c not in [outcome_col, 'gender']]
    X = df[embedding_cols]
    y = df[outcome_col]
    gender = df['gender']

    return X, y, gender


def cross_validate_simple(X, y, n_splits=5, random_state=42):
    """Simple cross-validation without the Regressions class"""
    # Create subject-level splits
    subjects = pd.Series(X.index.get_level_values(0).unique())
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Get predictions
    predictions = []
    true_values = []

    for train_idx, test_idx in kf.split(subjects):
        train_subjects = subjects.iloc[train_idx]
        test_subjects = subjects.iloc[test_idx]

        # Filter by subjects
        train_mask = X.index.get_level_values(0).isin(train_subjects)
        test_mask = X.index.get_level_values(0).isin(test_subjects)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        predictions.extend(y_pred)
        true_values.extend(y_test)

    # Calculate metrics
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    r2 = r2_score(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    spearman = spearmanr(true_values, predictions)[0]

    return {'r2': r2, 'mae': mae, 'spearman': spearman}

# ============================================================================
# 5. RUN CROSS-VALIDATION
# ============================================================================
print("\n[5/6] Running cross-validation...")

results = []

for category, outcomes in outcomes_to_test.items():
    print(f"\n{'='*60}")
    print(f"Category: {category.upper()}")
    print(f"{'='*60}")

    for outcome_col, system, outcome_name in outcomes:
        print(f"\n{outcome_name} ({outcome_col}):")

        # Test on each embedding type
        for emb_name, embeddings in [('DEXA', dexa_emb), ('Retina', retina_emb), ('Pooled', pooled_emb)]:
            try:
                # Prepare data
                data = prepare_dataset(embeddings, outcome_col, system)
                if data is None:
                    print(f"   {emb_name:8s}: No data")
                    continue

                X, y, gender = data

                # Cross-validate
                cv_result = cross_validate_simple(X, y)

                r2 = cv_result['r2']
                print(f"   {emb_name:8s}: n={len(X):5d}  R²={r2:6.3f}  Spearman={cv_result['spearman']:6.3f}")

                # Store result
                results.append({
                    'category': category,
                    'outcome': outcome_name,
                    'outcome_col': outcome_col,
                    'embedding_type': emb_name,
                    'n_samples': len(X),
                    'r2': r2,
                    'spearman': cv_result['spearman'],
                    'mae': cv_result['mae'],
                })

            except Exception as e:
                print(f"   {emb_name:8s}: ERROR - {str(e)}")
                continue

# ============================================================================
# 6. SAVE AND SUMMARIZE RESULTS
# ============================================================================
print("\n[6/6] Saving results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)

print(f"\n   Saved to: {OUTPUT_DIR / 'hypothesis_test_results.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS TEST SUMMARY")
print("=" * 80)

for category in outcomes_to_test.keys():
    cat_results = results_df[results_df['category'] == category]
    if len(cat_results) == 0:
        continue

    print(f"\n{category.upper().replace('_', ' ')}:")
    print("-" * 80)

    # Pivot table
    pivot = cat_results.pivot_table(
        index='outcome',
        columns='embedding_type',
        values='r2',
        aggfunc='mean'
    )

    # Format nicely
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Identify best embedding for each outcome
    print("\nBest embedding per outcome:")
    for outcome in pivot.index:
        row = pivot.loc[outcome]
        best_emb = row.idxmax()
        best_r2 = row.max()

        # Show all values for comparison
        values_str = "  ".join([f"{emb}:{row[emb]:.3f}" for emb in row.index])

        print(f"   {outcome:40s} → {best_emb:8s} (R²={best_r2:.3f})  [{values_str}]")

print("\n" + "=" * 80)
print("HYPOTHESIS TESTING COMPLETE")
print("=" * 80)
