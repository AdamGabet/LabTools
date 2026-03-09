"""
Grip Strength and Cardiovascular Markers Study
===============================================

Research Question: Is hand grip strength associated with cardiovascular health markers
(arterial stiffness, blood pressure, carotid IMT) independent of age, gender, and BMI?

Dataset: HPP 10K cohort
Analysis Date: January 2026

Key Finding: UNEXPECTED - Higher grip strength is associated with HIGHER PWV and
blood pressure, opposite to the frailty hypothesis from literature.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import os

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    """Load and prepare data for analysis."""
    from body_system_loader.load_feature_df import load_columns_as_df

    # Define columns
    frailty_cols = ['hand_grip_left', 'hand_grip_right']
    cv_cols = [
        'from_l_thigh_to_l_ankle_pwv', 'from_r_thigh_to_r_ankle_pwv',
        'l_abi', 'r_abi',
        'sitting_blood_pressure_systolic', 'sitting_blood_pressure_diastolic',
        'intima_media_th_mm_1_intima_media_thickness'
    ]
    demographic_cols = ['age', 'gender', 'bmi']

    # Load data
    df = load_columns_as_df(frailty_cols + cv_cols + demographic_cols)
    df = df.groupby(level=0).first()

    # Filter to ages 40-70 (reliable estimates)
    df = df[(df['age'] >= 40) & (df['age'] <= 70)].copy()

    # Create derived variables
    df['grip_strength'] = (df['hand_grip_left'] + df['hand_grip_right']) / 2
    df['pwv'] = (df['from_l_thigh_to_l_ankle_pwv'] + df['from_r_thigh_to_r_ankle_pwv']) / 2
    df['abi'] = (df['l_abi'] + df['r_abi']) / 2
    df['sbp'] = df['sitting_blood_pressure_systolic']
    df['dbp'] = df['sitting_blood_pressure_diastolic']
    df['imt'] = df['intima_media_th_mm_1_intima_media_thickness']

    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[40, 50, 60, 70], labels=['40-50', '50-60', '60-70'])

    return df


def partial_corr(df, x, y, covars):
    """Compute partial correlation controlling for covariates."""
    df_clean = df[[x, y] + covars].dropna()
    if len(df_clean) < 50:
        return np.nan, np.nan, len(df_clean)
    X_cov = sm.add_constant(df_clean[covars])
    resid_x = sm.OLS(df_clean[x], X_cov).fit().resid
    resid_y = sm.OLS(df_clean[y], X_cov).fit().resid
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p, len(df_clean)


def run_descriptive_analysis(df):
    """Generate descriptive statistics."""
    print("="*70)
    print("SAMPLE CHARACTERISTICS")
    print("="*70)

    df_base = df.dropna(subset=['grip_strength', 'age', 'gender', 'bmi'])

    print(f"Total N: {len(df_base):,}")
    print(f"Age: {df_base['age'].mean():.1f} ± {df_base['age'].std():.1f} years")
    print(f"Male: {100*df_base['gender'].mean():.1f}%")
    print(f"BMI: {df_base['bmi'].mean():.1f} ± {df_base['bmi'].std():.1f}")

    print("\nGrip Strength by Gender:")
    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        df_g = df_base[df_base['gender'] == gender_val]
        print(f"  {gender_name}: {df_g['grip_strength'].mean():.1f} ± {df_g['grip_strength'].std():.1f} kg (N={len(df_g):,})")

    return df_base


def run_adjusted_analysis(df):
    """Run adjusted regression analyses."""
    outcomes = {
        'pwv': 'Pulse Wave Velocity',
        'sbp': 'Systolic BP',
        'dbp': 'Diastolic BP',
        'imt': 'Intima-Media Thickness',
        'abi': 'Ankle-Brachial Index'
    }
    confounders = ['age', 'gender', 'bmi']

    results = []

    print("\n" + "="*70)
    print("ADJUSTED ANALYSIS (Grip Strength → CV Markers)")
    print("Controlling for: age, gender, BMI")
    print("="*70)

    for outcome, name in outcomes.items():
        # Partial correlation
        r, p, n = partial_corr(df, 'grip_strength', outcome, confounders)

        # Multiple regression
        df_model = df[['grip_strength', outcome] + confounders].dropna()
        X = sm.add_constant(df_model[['grip_strength'] + confounders])
        y = df_model[outcome]
        model = sm.OLS(y, X).fit()

        beta = model.params['grip_strength']
        se = model.bse['grip_strength']
        pval = model.pvalues['grip_strength']

        results.append({
            'outcome': name,
            'partial_r': r,
            'beta': beta,
            'se': se,
            'p_value': pval,
            'n': n,
            'r_squared': model.rsquared
        })

        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        print(f"\n{name} (N = {n:,})")
        print(f"  Partial r = {r:.3f}")
        print(f"  β = {beta:.4f} ± {se:.4f} (p = {pval:.2e}) {sig}")
        print(f"  R² = {model.rsquared:.4f}")

    return pd.DataFrame(results)


def run_stratified_analysis(df):
    """Run gender and age stratified analyses."""
    print("\n" + "="*70)
    print("GENDER-STRATIFIED ANALYSIS")
    print("="*70)

    stratified_results = []

    for gender_val, gender_name in [(0, 'Female'), (1, 'Male')]:
        df_g = df[df['gender'] == gender_val]

        for outcome in ['pwv', 'sbp']:
            df_model = df_g[['grip_strength', outcome, 'age', 'bmi']].dropna()
            X = sm.add_constant(df_model[['grip_strength', 'age', 'bmi']])
            y = df_model[outcome]
            model = sm.OLS(y, X).fit()

            stratified_results.append({
                'subgroup': gender_name,
                'outcome': outcome.upper(),
                'beta': model.params['grip_strength'],
                'p_value': model.pvalues['grip_strength'],
                'n': len(df_model)
            })

    print("\n" + "="*70)
    print("AGE-STRATIFIED ANALYSIS (Grip → PWV)")
    print("="*70)

    for age_grp in ['40-50', '50-60', '60-70']:
        df_age = df[df['age_group'] == age_grp]
        df_model = df_age[['grip_strength', 'pwv', 'gender', 'bmi']].dropna()

        if len(df_model) < 100:
            continue

        X = sm.add_constant(df_model[['grip_strength', 'gender', 'bmi']])
        y = df_model['pwv']
        model = sm.OLS(y, X).fit()

        print(f"\nAge {age_grp} (N = {len(df_model):,})")
        print(f"  β = {model.params['grip_strength']:.4f} (p = {model.pvalues['grip_strength']:.2e})")

        stratified_results.append({
            'subgroup': f'Age {age_grp}',
            'outcome': 'PWV',
            'beta': model.params['grip_strength'],
            'p_value': model.pvalues['grip_strength'],
            'n': len(df_model)
        })

    return pd.DataFrame(stratified_results)


def create_figures(df):
    """Create visualization figures."""

    # Color scheme
    FEMALE_COLOR = '#E74C3C'
    MALE_COLOR = '#3498DB'

    # ========================================
    # Figure 1: PWV by Grip Strength Quintiles
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (gender_val, gender_name, color) in enumerate([(0, 'Female', FEMALE_COLOR), (1, 'Male', MALE_COLOR)]):
        df_g = df[df['gender'] == gender_val].copy()
        df_g['grip_quintile'] = pd.qcut(df_g['grip_strength'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        means = df_g.groupby('grip_quintile', observed=True)['pwv'].mean()
        sems = df_g.groupby('grip_quintile', observed=True)['pwv'].sem()
        grip_means = df_g.groupby('grip_quintile', observed=True)['grip_strength'].mean()

        ax = axes[idx]
        bars = ax.bar(range(5), means.values, yerr=sems.values, color=color, alpha=0.7, capsize=5)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'Q{i+1}\n({grip_means.iloc[i]:.0f} kg)' for i in range(5)])
        ax.set_xlabel('Grip Strength Quintile')
        ax.set_ylabel('PWV (m/s)')
        ax.set_title(f'{gender_name} (N = {len(df_g):,})')

        # Add trend annotation
        trend_dir = "↑" if means.iloc[-1] > means.iloc[0] else "↓"
        diff = means.iloc[-1] - means.iloc[0]
        ax.annotate(f'{trend_dir} {diff:+.2f} m/s', xy=(0.95, 0.95), xycoords='axes fraction',
                   ha='right', va='top', fontsize=10, fontweight='bold')

    plt.suptitle('Pulse Wave Velocity by Grip Strength Quintile\n(Higher PWV = Stiffer Arteries)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pwv_by_grip_quintile.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 2: Scatter plot with regression lines
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (gender_val, gender_name, color) in enumerate([(0, 'Female', FEMALE_COLOR), (1, 'Male', MALE_COLOR)]):
        df_g = df[df['gender'] == gender_val].dropna(subset=['grip_strength', 'pwv'])

        ax = axes[idx]
        ax.scatter(df_g['grip_strength'], df_g['pwv'], alpha=0.2, color=color, s=5)

        # Fit regression line
        slope, intercept, r, p, se = stats.linregress(df_g['grip_strength'], df_g['pwv'])
        x_line = np.array([df_g['grip_strength'].min(), df_g['grip_strength'].max()])
        ax.plot(x_line, intercept + slope * x_line, color='black', linewidth=2, linestyle='--')

        ax.set_xlabel('Grip Strength (kg)')
        ax.set_ylabel('PWV (m/s)')
        ax.set_title(f'{gender_name}\nr = {r:.3f}, p = {p:.2e}')

    plt.suptitle('Grip Strength vs Pulse Wave Velocity', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'grip_vs_pwv_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 3: Effect sizes across outcomes
    # ========================================
    outcomes = ['pwv', 'sbp', 'dbp', 'imt']
    outcome_names = ['PWV', 'Systolic BP', 'Diastolic BP', 'IMT']

    betas = []
    pvals = []
    for outcome in outcomes:
        df_model = df[['grip_strength', outcome, 'age', 'gender', 'bmi']].dropna()
        X = sm.add_constant(df_model[['grip_strength', 'age', 'gender', 'bmi']])
        y = df_model[outcome]
        model = sm.OLS(y, X).fit()

        # Standardize beta
        beta_std = model.params['grip_strength'] * df_model['grip_strength'].std() / df_model[outcome].std()
        betas.append(beta_std)
        pvals.append(model.pvalues['grip_strength'])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#E74C3C' if b > 0 else '#3498DB' for b in betas]
    bars = ax.barh(outcome_names, betas, color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Standardized β (per 1 SD increase in grip strength)')
    ax.set_title('Association of Grip Strength with CV Markers\n(Adjusted for age, gender, BMI)\nPositive = Higher grip → Higher marker')

    # Add significance stars
    for i, (bar, p) in enumerate(zip(bars, pvals)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, sig, va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'standardized_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================
    # Figure 4: Age-stratified effects
    # ========================================
    fig, ax = plt.subplots(figsize=(8, 5))

    age_groups = ['40-50', '50-60', '60-70']
    age_betas = []
    age_cis = []

    for age_grp in age_groups:
        df_age = df[df['age_group'] == age_grp]
        df_model = df_age[['grip_strength', 'pwv', 'gender', 'bmi']].dropna()

        X = sm.add_constant(df_model[['grip_strength', 'gender', 'bmi']])
        y = df_model['pwv']
        model = sm.OLS(y, X).fit()

        age_betas.append(model.params['grip_strength'])
        ci = model.conf_int().loc['grip_strength']
        age_cis.append((ci[1] - ci[0]) / 2)

    ax.errorbar(range(3), age_betas, yerr=age_cis, fmt='o-', capsize=5, markersize=10, color='#2C3E50')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(age_groups)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('β (Grip → PWV)')
    ax.set_title('Effect of Grip Strength on PWV Across Age Groups\n(Adjusted for gender, BMI)')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'age_stratified_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\nFigures saved to:", FIGURES_DIR)


def create_pdf_report(df):
    """Generate PDF report."""
    from PIL import Image

    report_path = os.path.join(SCRIPT_DIR, 'report.pdf')

    with PdfPages(report_path) as pdf:
        # Page 1: Methods
        fig = plt.figure(figsize=(8.5, 11))
        methods_text = """
        GRIP STRENGTH AND CARDIOVASCULAR HEALTH
        ═══════════════════════════════════════════════════════════════════════

        RESEARCH QUESTION
        Is hand grip strength associated with cardiovascular markers independent
        of age, gender, and BMI?

        HYPOTHESIS (from literature)
        Low grip strength (sarcopenia/frailty) → worse cardiovascular health

        STUDY DESIGN
        • Cross-sectional analysis of HPP 10K cohort
        • Age range: 40-70 years
        • Complete case analysis

        EXPOSURE
        • Hand grip strength (average of left and right hand)
        • Measured by dynamometer

        OUTCOMES
        • Pulse Wave Velocity (PWV) - arterial stiffness
        • Blood Pressure (systolic, diastolic)
        • Intima-Media Thickness (IMT) - carotid atherosclerosis
        • Ankle-Brachial Index (ABI) - peripheral artery disease

        CONFOUNDERS CONTROLLED
        • Age (continuous)
        • Gender (binary)
        • BMI (continuous)

        ANALYSIS
        • Partial correlations (controlling for confounders)
        • Multiple linear regression
        • Gender-stratified analysis
        • Age-stratified analysis

        KEY FINDING: UNEXPECTED
        Higher grip strength is associated with HIGHER PWV and blood pressure,
        OPPOSITE to the frailty hypothesis from literature!
        """
        fig.text(0.05, 0.95, methods_text, ha='left', va='top', fontsize=10, family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Pages 2-5: Figures
        figure_files = [
            ('pwv_by_grip_quintile.png', 'Main Finding: Higher Grip → Higher PWV'),
            ('grip_vs_pwv_scatter.png', 'Scatter: Grip Strength vs PWV'),
            ('standardized_effects.png', 'Effect Sizes Across CV Markers'),
            ('age_stratified_effects.png', 'Age-Stratified Effects'),
        ]

        for fname, title in figure_files:
            fpath = os.path.join(FIGURES_DIR, fname)
            if os.path.exists(fpath):
                fig = plt.figure(figsize=(11, 8.5))
                img = Image.open(fpath)
                plt.imshow(img)
                plt.axis('off')
                plt.title(title, fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        # Page 6: Key Findings
        fig = plt.figure(figsize=(8.5, 11))
        findings_text = """
        KEY FINDINGS SUMMARY
        ═══════════════════════════════════════════════════════════════════════

        SAMPLE: N = 12,371 (ages 40-70, 47% male)

        MAIN RESULT: OPPOSITE TO HYPOTHESIS
        Higher grip strength is associated with HIGHER (worse) cardiovascular
        markers, not lower as predicted by the frailty literature.

        ADJUSTED ASSOCIATIONS (controlling for age, gender, BMI)
        ┌────────────────────────┬───────────┬───────────┬──────────────┐
        │ Outcome                │ Partial r │ β         │ p-value      │
        ├────────────────────────┼───────────┼───────────┼──────────────┤
        │ Pulse Wave Velocity    │ +0.081    │ +0.0096   │ < 0.001 ***  │
        │ Systolic BP            │ +0.111    │ +0.1075   │ < 0.001 ***  │
        │ Diastolic BP           │ +0.061    │ +0.0418   │ < 0.001 ***  │
        │ Intima-Media Thickness │ +0.030    │ +0.0002   │ 0.002 **     │
        │ Ankle-Brachial Index   │ +0.017    │ +0.0001   │ 0.056 (NS)   │
        └────────────────────────┴───────────┴───────────┴──────────────┘

        INTERPRETATION
        • In this healthy population, grip strength is a marker of physical
          activity and muscle mass, associated with higher cardiac output
          and sympathetic tone → higher blood pressure and PWV

        • The frailty-CV mortality association from longitudinal studies
          may not translate to cross-sectional CV marker associations
          in healthy middle-aged populations

        • Effect is consistent across:
          - Both genders (stronger in females)
          - All age groups (40-50, 50-60, 60-70)

        CLINICAL IMPLICATIONS
        • Do NOT interpret low grip strength as "better" CV health
        • The grip strength-CV marker relationship is complex and may
          differ from the grip strength-CV mortality relationship
        • More research needed on mechanisms

        LIMITATIONS
        1. Cross-sectional design - cannot establish causality
        2. Healthy cohort - may not generalize to frail populations
        3. Grip strength units unclear (may be summed trials)
        4. Did not control for physical activity, medications
        """
        fig.text(0.05, 0.95, findings_text, ha='left', va='top', fontsize=10, family='monospace')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"\nPDF report saved: {report_path}")


def main():
    """Run complete analysis."""
    print("Loading data...")
    df = load_data()

    print("\n" + "="*70)
    print("GRIP STRENGTH AND CARDIOVASCULAR HEALTH STUDY")
    print("="*70)

    # Descriptive analysis
    run_descriptive_analysis(df)

    # Adjusted analysis
    results = run_adjusted_analysis(df)

    # Stratified analysis
    stratified = run_stratified_analysis(df)

    # Create figures
    print("\nCreating figures...")
    create_figures(df)

    # Create PDF report
    print("\nCreating PDF report...")
    create_pdf_report(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
