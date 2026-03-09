"""
COMPREHENSIVE hypothesis testing with many more outcomes and both Spearman + Pearson correlations
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/adamgab/PycharmProjects/LabTools')

from body_system_loader.load_feature_df import load_body_system_df
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
EMBED_DIR = Path("/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh")
OUTPUT_DIR = Path("/home/adamgab/PycharmProjects/LabTools/research/jepa_modality_fusion_hypothesis")

print("=" * 80)
print("COMPREHENSIVE JEPA HYPOTHESIS TESTING")
print("=" * 80)

# ============================================================================
# 1. LOAD EMBEDDINGS
# ============================================================================
print("\n[1/5] Loading embeddings...")
dexa_emb = pd.read_csv(EMBED_DIR / "dexa_embeddings.csv").set_index(['RegistrationCode', 'research_stage'])
retina_emb = pd.read_csv(EMBED_DIR / "retina_embeddings.csv").set_index(['RegistrationCode', 'research_stage'])
pooled_emb = pd.read_csv(EMBED_DIR / "pooled_embeddings.csv").set_index(['RegistrationCode', 'research_stage'])
pooled_emb = pooled_emb[~pooled_emb.index.duplicated(keep='first')]

print(f"   DEXA: {dexa_emb.shape}")
print(f"   Retina: {retina_emb.shape}")
print(f"   Pooled: {pooled_emb.shape}")

# ============================================================================
# 2. EXPANDED OUTCOMES - WAY MORE LABELS
# ============================================================================
print("\n[2/5] Loading expanded outcome set...")

# Load all relevant body systems
age_gender_bmi = load_body_system_df('Age_Gender_BMI')
body_comp = load_body_system_df('body_composition')
cardio = load_body_system_df('cardiovascular_system')
glycemic = load_body_system_df('glycemic_status')
immune = load_body_system_df('immune_system')
lipids = load_body_system_df('blood_lipids')
liver = load_body_system_df('liver')
renal = load_body_system_df('renal_function')
bone_density = load_body_system_df('bone_density')
frailty = load_body_system_df('frailty')

print(f"   Loaded {10} body systems")

# MASSIVELY EXPANDED outcome list
outcomes_to_test = {
    'shared': [
        ('age', age_gender_bmi, 'Age'),
        ('bmi', age_gender_bmi, 'BMI'),
    ],

    'dexa_specific': [
        # Lean mass & muscle
        ('body_comp_total_lean_mass', body_comp, 'Total Lean Mass'),
        ('body_comp_left_arm_lean_mass', body_comp, 'Left Arm Lean Mass'),
        ('body_comp_right_arm_lean_mass', body_comp, 'Right Arm Lean Mass'),
        ('body_comp_left_leg_lean_mass', body_comp, 'Left Leg Lean Mass'),
        ('body_comp_right_leg_lean_mass', body_comp, 'Right Leg Lean Mass'),
        ('body_comp_trunk_lean_mass', body_comp, 'Trunk Lean Mass'),

        # Fat mass & distribution
        ('body_comp_total_fat_mass', body_comp, 'Total Fat Mass'),
        ('body_comp_android_fat_mass', body_comp, 'Android Fat Mass'),
        ('body_comp_gynoid_fat_mass', body_comp, 'Gynoid Fat Mass'),
        ('total_scan_vat_mass', body_comp, 'Visceral Fat Mass'),
        ('total_scan_sat_mass', body_comp, 'Subcutaneous Fat Mass'),
        ('total_scan_vat_volume', body_comp, 'Visceral Fat Volume'),
        ('total_scan_sat_volume', body_comp, 'Subcutaneous Fat Volume'),

        # Body measurements
        ('weight', body_comp, 'Weight'),
        ('height', body_comp, 'Height'),
        ('waist', body_comp, 'Waist Circumference'),
        ('hips', body_comp, 'Hip Circumference'),

        # Bone mass (from body_comp)
        ('body_comp_total_bone_mass', body_comp, 'Total Bone Mass'),
        ('body_comp_left_arm_bone_mass', body_comp, 'Left Arm Bone Mass'),
        ('body_comp_right_arm_bone_mass', body_comp, 'Right Arm Bone Mass'),
        ('body_comp_left_leg_bone_mass', body_comp, 'Left Leg Bone Mass'),
        ('body_comp_right_leg_bone_mass', body_comp, 'Right Leg Bone Mass'),
    ],

    'retina_specific': [
        # Retinal vessel morphology
        ('automorph_artery_average_width', cardio, 'Artery Average Width'),
        ('automorph_vein_average_width', cardio, 'Vein Average Width'),
        ('automorph_artery_fractal_dimension', cardio, 'Artery Fractal Dimension'),
        ('automorph_vein_fractal_dimension', cardio, 'Vein Fractal Dimension'),
        ('automorph_fractal_dimension', cardio, 'Overall Fractal Dimension'),

        # Tortuosity
        ('automorph_artery_tortuosity_density', cardio, 'Artery Tortuosity Density'),
        ('automorph_vein_tortuosity_density', cardio, 'Vein Tortuosity Density'),
        ('automorph_artery_distance_tortuosity', cardio, 'Artery Distance Tortuosity'),
        ('automorph_vein_distance_tortuosity', cardio, 'Vein Distance Tortuosity'),
        ('automorph_artery_squared_curvature_tortuosity', cardio, 'Artery Curvature Tortuosity'),
        ('automorph_vein_squared_curvature_tortuosity', cardio, 'Vein Curvature Tortuosity'),

        # Vessel density
        ('automorph_artery_vessel_density', cardio, 'Artery Vessel Density'),
        ('automorph_vein_vessel_density', cardio, 'Vein Vessel Density'),

        # Cup/disc measurements
        ('automorph_cdr_horizontal', cardio, 'CDR Horizontal'),
        ('automorph_cdr_vertical', cardio, 'CDR Vertical'),
        ('automorph_disc_width', cardio, 'Optic Disc Width'),
        ('automorph_disc_height', cardio, 'Optic Disc Height'),
    ],

    'cross_modal': [
        # Glycemic
        ('bt__glucose', glycemic, 'Glucose'),
        ('bt__hba1c', glycemic, 'HbA1c'),
        ('iglu_cv', glycemic, 'CGM Coefficient of Variation'),
        ('iglu_gmi', glycemic, 'Glucose Management Indicator'),
        ('iglu_auc', glycemic, 'CGM Area Under Curve'),

        # Immune/inflammation
        ('bt__wbc', immune, 'White Blood Cells'),
        ('bt__neutrophils_abs', immune, 'Neutrophils'),
        ('bt__lymphocytes_abs', immune, 'Lymphocytes'),
        ('bt__monocytes_abs', immune, 'Monocytes'),

        # Lipids (if available)
        ('bt__triglycerides', lipids, 'Triglycerides'),
        ('bt__hdl_cholesterol', lipids, 'HDL Cholesterol'),
        ('bt__ldl_cholesterol', lipids, 'LDL Cholesterol'),
        ('bt__total_cholesterol', lipids, 'Total Cholesterol'),

        # Liver
        ('bt__alt', liver, 'ALT'),
        ('bt__ast', liver, 'AST'),
        ('bt__alp', liver, 'Alkaline Phosphatase'),
        ('bt__ggt', liver, 'GGT'),

        # Renal
        ('bt__creatinine', renal, 'Creatinine'),
        ('bt__urea', renal, 'Urea'),
        ('egfr', renal, 'eGFR'),
    ]
}

total_outcomes = sum(len(v) for v in outcomes_to_test.values())
print(f"   Testing {total_outcomes} outcomes across {len(outcomes_to_test)} categories")

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def prepare_dataset(embeddings, outcome_col, outcome_df):
    """Merge embeddings with outcome + gender"""
    if outcome_col not in outcome_df.columns:
        return None

    df = embeddings.join(outcome_df[[outcome_col]], how='inner')
    df = df.join(age_gender_bmi[['gender']], how='inner')
    df = df.dropna(subset=[outcome_col, 'gender'])

    if len(df) == 0:
        return None

    embedding_cols = [c for c in df.columns if c not in [outcome_col, 'gender']]
    X = df[embedding_cols]
    y = df[outcome_col]

    return X, y

def cross_validate_comprehensive(X, y, n_splits=5, random_state=42):
    """Cross-validation with Spearman AND Pearson correlations"""
    subjects = pd.Series(X.index.get_level_values(0).unique())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    predictions = []
    true_values = []

    for train_idx, test_idx in kf.split(subjects):
        train_subjects = subjects.iloc[train_idx]
        test_subjects = subjects.iloc[test_idx]

        train_mask = X.index.get_level_values(0).isin(train_subjects)
        test_mask = X.index.get_level_values(0).isin(test_subjects)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predictions.extend(y_pred)
        true_values.extend(y_test)

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    r2 = r2_score(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    spearman = spearmanr(true_values, predictions)[0]
    pearson = pearsonr(true_values, predictions)[0]

    return {
        'r2': r2,
        'mae': mae,
        'spearman': spearman,
        'pearson': pearson
    }

# ============================================================================
# 4. RUN COMPREHENSIVE CROSS-VALIDATION
# ============================================================================
print("\n[4/5] Running comprehensive cross-validation...")

results = []

for category, outcomes in outcomes_to_test.items():
    print(f"\n{'='*70}")
    print(f"{category.upper().replace('_', ' ')}")
    print(f"{'='*70}")

    for outcome_col, outcome_df, outcome_name in outcomes:
        if len(results) % 5 == 0:
            print(f"\nTested {len(results)}/{total_outcomes} outcomes...")

        for emb_name, embeddings in [('DEXA', dexa_emb), ('Retina', retina_emb), ('Pooled', pooled_emb)]:
            try:
                data = prepare_dataset(embeddings, outcome_col, outcome_df)
                if data is None:
                    continue

                X, y = data
                cv_result = cross_validate_comprehensive(X, y)

                results.append({
                    'category': category,
                    'outcome': outcome_name,
                    'outcome_col': outcome_col,
                    'embedding_type': emb_name,
                    'n_samples': len(X),
                    'r2': cv_result['r2'],
                    'spearman': cv_result['spearman'],
                    'pearson': cv_result['pearson'],
                    'mae': cv_result['mae'],
                })

            except Exception as e:
                continue

print(f"\nCompleted {len(results)} total tests!")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
print("\n[5/5] Saving comprehensive results...")

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "comprehensive_results.csv", index=False)

print(f"   Saved to: {OUTPUT_DIR / 'comprehensive_results.csv'}")
print(f"   Total successful tests: {len(results_df)}")
print(f"   Outcomes tested: {results_df['outcome'].nunique()}")

# Quick summary
print("\n" + "=" * 80)
print("QUICK SUMMARY")
print("=" * 80)

for category in outcomes_to_test.keys():
    cat_results = results_df[results_df['category'] == category]
    if len(cat_results) == 0:
        continue

    print(f"\n{category.upper().replace('_', ' ')}:")

    # Count best per metric
    pivot_r2 = cat_results.pivot_table(index='outcome', columns='embedding_type', values='r2')
    pivot_spear = cat_results.pivot_table(index='outcome', columns='embedding_type', values='spearman')
    pivot_pears = cat_results.pivot_table(index='outcome', columns='embedding_type', values='pearson')

    best_r2 = pivot_r2.idxmax(axis=1).value_counts()
    best_spear = pivot_spear.idxmax(axis=1).value_counts()
    best_pears = pivot_pears.idxmax(axis=1).value_counts()

    print(f"   Best by R²: {dict(best_r2)}")
    print(f"   Best by Spearman: {dict(best_spear)}")
    print(f"   Best by Pearson: {dict(best_pears)}")

print("\n" + "=" * 80)
print("COMPREHENSIVE TESTING COMPLETE")
print("=" * 80)
