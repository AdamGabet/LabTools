#!/usr/bin/env python
"""
Gender-Stratified Prevalence Report
====================================
Shows prevalence ranges (40-70 years) for each condition, stratified by gender.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUTPUT_DIR = '/home/adamgab/PycharmProjects/LabTools/research/age_medical_conditions_analysis/gender_stratified'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disease groupings (same as main analysis)
DISEASE_GROUPS = {
    'Cardiovascular': [
        'Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
        'Heart valve disease', 'AV. Conduction Disorder', 'Myocarditis',
        'Atherosclerotic'
    ],
    'Cancer': [
        'Breast Cancer', 'Melanoma', 'Lymphoma'
    ],
    'Respiratory': [
        'Asthma', 'COPD'
    ],
    'Metabolic_Endocrine': [
        'Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Hypercholesterolaemia',
        'Obesity', 'Fatty Liver Disease', 'Hashimoto', 'Goiter',
        'Hyperparathyroidism', 'Thyroid Adenoma', 'G6PD'
    ],
    'Mental_Health': [
        'Depression', 'Anxiety', 'ADHD', 'PTSD', 'Insomnia',
        'Migraine', 'Headache'
    ],
    'Musculoskeletal': [
        'Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures',
        'Gout', 'Meniscus Tears'
    ],
    'Gastrointestinal': [
        'IBD', 'IBS', 'Celiac', 'Peptic Ulcer Disease', 'Gallstone Disease',
        'Haemorrhoids', 'Anal Fissure', 'Anal abscess'
    ],
    'Autoimmune': [
        'Psoriasis', 'Vitiligo', 'Hashimoto', 'Allergy', 'Atopic Dermatitis',
        'Uveitis', 'FMF'
    ],
    'Eye_Ear_Sensory': [
        'Glaucoma', 'Hearing loss', 'Tinnitus', 'Retinal detachment',
        'Ocular Hypertension'
    ],
    'Renal_Urological': [
        'Renal Stones', 'Urinary Tract Stones', 'Urinary tract infection'
    ]
}


def get_condition_to_group_mapping():
    """Create mapping from condition to disease group."""
    mapping = {}
    for group, conditions in DISEASE_GROUPS.items():
        for condition in conditions:
            if condition not in mapping:
                mapping[condition] = group
    return mapping


def calculate_prevalence_range_by_gender(df, condition, gender_val, age_range=(40, 70)):
    """Calculate prevalence at start and end of age range for one gender."""
    d = df[(df['gender'] == gender_val) & 
           (df['age'] >= age_range[0]) & 
           (df['age'] <= age_range[1])].copy()
    
    if len(d) < 50:
        return None
    
    # Calculate prevalence at age 40 and 70 using 5-year windows
    age_40_window = d[(d['age'] >= 40) & (d['age'] < 45)]
    age_70_window = d[(d['age'] >= 65) & (d['age'] <= 70)]
    
    if len(age_40_window) < 20 or len(age_70_window) < 20:
        return None
    
    prev_40 = 100 * age_40_window[condition].sum() / len(age_40_window)
    prev_70 = 100 * age_70_window[condition].sum() / len(age_70_window)
    
    n_total = len(d)
    n_positive = int(d[condition].sum())
    
    return {
        'prev_40': prev_40,
        'prev_70': prev_70,
        'n_total': n_total,
        'n_positive': n_positive,
        'change': prev_70 - prev_40,
        'fold_change': prev_70 / prev_40 if prev_40 > 0 else np.nan
    }


def analyze_all_conditions_by_gender(df):
    """Analyze prevalence ranges for all conditions by gender."""
    condition_cols = [c for c in df.columns if c not in ['age', 'gender']]
    
    results = []
    for condition in condition_cols:
        # Male analysis
        male_stats = calculate_prevalence_range_by_gender(df, condition, 1)
        
        # Female analysis
        female_stats = calculate_prevalence_range_by_gender(df, condition, 0)
        
        # Only include if we have data for at least one gender
        if male_stats is None and female_stats is None:
            continue
        
        # Get disease group
        mapping = get_condition_to_group_mapping()
        disease_group = mapping.get(condition, 'Other')
        
        result = {
            'condition': condition,
            'disease_group': disease_group,
        }
        
        # Add male stats
        if male_stats:
            result.update({
                'male_prev_40': male_stats['prev_40'],
                'male_prev_70': male_stats['prev_70'],
                'male_change': male_stats['change'],
                'male_n_positive': male_stats['n_positive'],
                'male_n_total': male_stats['n_total'],
            })
        else:
            result.update({
                'male_prev_40': np.nan,
                'male_prev_70': np.nan,
                'male_change': np.nan,
                'male_n_positive': 0,
                'male_n_total': 0,
            })
        
        # Add female stats
        if female_stats:
            result.update({
                'female_prev_40': female_stats['prev_40'],
                'female_prev_70': female_stats['prev_70'],
                'female_change': female_stats['change'],
                'female_n_positive': female_stats['n_positive'],
                'female_n_total': female_stats['n_total'],
            })
        else:
            result.update({
                'female_prev_40': np.nan,
                'female_prev_70': np.nan,
                'female_change': np.nan,
                'female_n_positive': 0,
                'female_n_total': 0,
            })
        
        results.append(result)
    
    return pd.DataFrame(results)


def generate_markdown_report(df, results_df):
    """Generate comprehensive markdown report with prevalence ranges."""
    
    n_male = (df['gender'] == 1).sum()
    n_female = (df['gender'] == 0).sum()
    
    report = f"""# Gender-Stratified Prevalence Report
## HPP 10K Dataset (Ages 40-70)

### Executive Summary

This report shows the prevalence of medical conditions at ages 40 and 70, 
stratified by gender. Prevalence is shown as percentage (%) of the population 
with each condition.

**Sample Sizes:**
- **Male**: {n_male:,} subjects
- **Female**: {n_female:,} subjects
- **Age range**: 40-70 years

---

## Interpretation Guide

- **Prev@40**: Prevalence (%) at age 40 (using 40-45 age window)
- **Prev@70**: Prevalence (%) at age 70 (using 65-70 age window)
- **Change**: Absolute change in prevalence (percentage points)
- **N**: Number of subjects with the condition

**Color coding in tables:**
- 🔴 Red: Large increase with age (>5 percentage points)
- 🟡 Yellow: Moderate change (2-5 percentage points)
- 🟢 Green: Minimal change (<2 percentage points)
- 🔵 Blue: Decrease with age

---

## Summary Statistics

"""
    
    # Count conditions by change pattern
    male_increase = results_df[results_df['male_change'] > 2].shape[0]
    male_stable = results_df[(results_df['male_change'] >= -2) & (results_df['male_change'] <= 2)].shape[0]
    male_decrease = results_df[results_df['male_change'] < -2].shape[0]
    
    female_increase = results_df[results_df['female_change'] > 2].shape[0]
    female_stable = results_df[(results_df['female_change'] >= -2) & (results_df['female_change'] <= 2)].shape[0]
    female_decrease = results_df[results_df['female_change'] < -2].shape[0]
    
    report += f"""
**Males:**
- {male_increase} conditions increase substantially (>2pp)
- {male_stable} conditions remain stable (±2pp)
- {male_decrease} conditions decrease (>2pp)

**Females:**
- {female_increase} conditions increase substantially (>2pp)
- {female_stable} conditions remain stable (±2pp)
- {female_decrease} conditions decrease (>2pp)

---

## Top Conditions by Prevalence Change

### Males: Largest Increases with Age

"""
    
    # Top male increases
    male_top = results_df[results_df['male_n_positive'] >= 50].nlargest(15, 'male_change')
    
    report += "| Condition | Prev@40 | Prev@70 | Change | N Cases |\n"
    report += "|-----------|---------|---------|--------|----------|\n"
    
    for _, row in male_top.iterrows():
        report += f"| {row['condition']} | {row['male_prev_40']:.2f}% | {row['male_prev_70']:.2f}% | "
        report += f"+{row['male_change']:.2f}pp | {row['male_n_positive']:,} |\n"
    
    report += "\n### Females: Largest Increases with Age\n\n"
    
    # Top female increases
    female_top = results_df[results_df['female_n_positive'] >= 50].nlargest(15, 'female_change')
    
    report += "| Condition | Prev@40 | Prev@70 | Change | N Cases |\n"
    report += "|-----------|---------|---------|--------|----------|\n"
    
    for _, row in female_top.iterrows():
        report += f"| {row['condition']} | {row['female_prev_40']:.2f}% | {row['female_prev_70']:.2f}% | "
        report += f"+{row['female_change']:.2f}pp | {row['female_n_positive']:,} |\n"
    
    report += "\n---\n\n## Conditions with Largest Gender Differences\n\n"
    
    # Calculate gender difference in change
    results_df['gender_diff'] = abs(results_df['male_change'] - results_df['female_change'])
    gender_diff_top = results_df.dropna(subset=['gender_diff']).nlargest(15, 'gender_diff')
    
    report += "| Condition | Male Change | Female Change | Difference |\n"
    report += "|-----------|-------------|---------------|------------|\n"
    
    for _, row in gender_diff_top.iterrows():
        report += f"| {row['condition']} | {row['male_change']:+.2f}pp | {row['female_change']:+.2f}pp | "
        report += f"{row['gender_diff']:.2f}pp |\n"
    
    report += "\n---\n\n## Complete Results by Disease Category\n\n"
    
    # Results by disease group
    for group in sorted(DISEASE_GROUPS.keys()):
        group_data = results_df[results_df['disease_group'] == group].copy()
        if len(group_data) == 0:
            continue
        
        # Sort by total prevalence (average of male and female at age 70)
        group_data['avg_prev_70'] = (group_data['male_prev_70'] + group_data['female_prev_70']) / 2
        group_data = group_data.sort_values('avg_prev_70', ascending=False)
        
        report += f"\n### {group.replace('_', ' ')}\n\n"
        report += "| Condition | Male 40→70 | Female 40→70 | Male N | Female N |\n"
        report += "|-----------|------------|--------------|--------|----------|\n"
        
        for _, row in group_data.iterrows():
            male_str = f"{row['male_prev_40']:.1f}→{row['male_prev_70']:.1f}%" if not np.isnan(row['male_prev_40']) else "N/A"
            female_str = f"{row['female_prev_40']:.1f}→{row['female_prev_70']:.1f}%" if not np.isnan(row['female_prev_40']) else "N/A"
            
            report += f"| {row['condition']} | {male_str} | {female_str} | "
            report += f"{row['male_n_positive']:,} | {row['female_n_positive']:,} |\n"
    
    report += """
---

## Methodology

### Data Source
- HPP 10K Dataset
- Age range: 40-70 years (where data is most reliable)
- Male: {n_male:,} subjects
- Female: {n_female:,} subjects

### Prevalence Calculation
- **Prev@40**: Prevalence in 40-45 age window
- **Prev@70**: Prevalence in 65-70 age window
- **Change**: Absolute difference in prevalence (percentage points)

### Inclusion Criteria
- Minimum 50 cases in at least one gender
- Minimum 20 subjects in each age window

### Limitations
1. Cross-sectional design (cannot establish causality)
2. Survival bias may affect older age groups
3. Detection bias (diagnosis rates may vary by age/gender)
4. Self-reported conditions may have recall bias
5. Some conditions are sex-specific (e.g., breast cancer, PCOS)

---

*Report generated from HPP 10K dataset*
""".format(n_male=n_male, n_female=n_female)
    
    return report


def main():
    """Main analysis pipeline."""
    print("=" * 60)
    print("GENDER-STRATIFIED PREVALENCE REPORT")
    print("HPP 10K Dataset (Ages 40-70)")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
    mc_df = load_body_system_df('medical_conditions')
    df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
    
    # Filter to age 40-70
    df = df[(df['age'] >= 40) & (df['age'] <= 70)]
    
    print(f"   Total subjects: {len(df):,}")
    print(f"   Male: {(df['gender']==1).sum():,}")
    print(f"   Female: {(df['gender']==0).sum():,}")
    
    # Analyze conditions
    print("\n2. Analyzing prevalence ranges by gender...")
    results_df = analyze_all_conditions_by_gender(df)
    
    # Save results
    csv_path = os.path.join(OUTPUT_DIR, 'gender_prevalence_ranges.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"   Saved results to: {csv_path}")
    
    # Generate report
    print("\n3. Generating markdown report...")
    report = generate_markdown_report(df, results_df)
    
    report_path = os.path.join(OUTPUT_DIR, 'GENDER_PREVALENCE_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   Saved report to: {report_path}")
    
    print("\n" + "=" * 60)
    print("REPORT COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\nKey Findings:")
    print(f"  - Analyzed {len(results_df)} conditions")
    
    # Top male increases
    male_top = results_df[results_df['male_n_positive'] >= 50].nlargest(3, 'male_change')
    print("\n  Top 3 Male increases:")
    for _, row in male_top.iterrows():
        print(f"    • {row['condition']}: {row['male_prev_40']:.1f}% → {row['male_prev_70']:.1f}% (+{row['male_change']:.1f}pp)")
    
    # Top female increases
    female_top = results_df[results_df['female_n_positive'] >= 50].nlargest(3, 'female_change')
    print("\n  Top 3 Female increases:")
    for _, row in female_top.iterrows():
        print(f"    • {row['condition']}: {row['female_prev_40']:.1f}% → {row['female_prev_70']:.1f}% (+{row['female_change']:.1f}pp)")
    
    return df, results_df


if __name__ == "__main__":
    df, results_df = main()
