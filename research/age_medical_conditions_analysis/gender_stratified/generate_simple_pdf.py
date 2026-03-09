#!/usr/bin/env python
"""Generate simple gender-stratified PDF report - NO FITTED LINES."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
from PIL import Image

OUT = '/home/adamgab/PycharmProjects/LabTools/research/age_medical_conditions_analysis/gender_stratified'

def add_image_page(pdf, image_path, title=None):
    """Add an image page to PDF."""
    if not os.path.exists(image_path):
        print(f"  Warning: {image_path} not found")
        return False
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    return True

def add_text_page(pdf, content, fontsize=10):
    """Add a text page to PDF."""
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, content, ha='left', va='top', fontsize=fontsize,
             family='monospace', wrap=True, transform=fig.transFigure)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print("Generating simple gender-stratified PDF report...")

# Load results
results = pd.read_csv(os.path.join(OUT, 'gender_prevalence_ranges.csv'))
print(f"Loaded {len(results)} conditions")

pdf_path = os.path.join(OUT, 'Gender_Stratified_Simple_Prevalence_Report.pdf')

with PdfPages(pdf_path) as pdf:

    # ===== TITLE PAGE =====
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.65, 'Age and Medical Conditions', ha='center', fontsize=32, fontweight='bold')
    fig.text(0.5, 0.55, 'Gender-Stratified Prevalence', ha='center', fontsize=28, fontweight='bold', color='#8E44AD')
    fig.text(0.5, 0.45, 'Raw Data - No Fitted Lines', ha='center', fontsize=20, color='#E74C3C')
    fig.text(0.5, 0.35, 'HPP 10K Dataset', ha='center', fontsize=20)
    fig.text(0.5, 0.28, 'Ages 40-70', ha='center', fontsize=16)
    fig.text(0.5, 0.21, 'Female: 12,313  |  Male: 11,152', ha='center', fontsize=14)
    fig.text(0.5, 0.11, 'Blue = Male  |  Red = Female', ha='center', fontsize=14, fontweight='bold')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)
    print("  Added: Title page")

    # ===== EXECUTIVE SUMMARY =====
    # Find conditions with large changes
    results['male_change_abs'] = results['male_change'].abs()
    results['female_change_abs'] = results['female_change'].abs()
    
    male_top = results[results['male_n_positive'] >= 50].nlargest(5, 'male_change')
    female_top = results[results['female_n_positive'] >= 50].nlargest(5, 'female_change')

    summary = f"""
EXECUTIVE SUMMARY: GENDER-STRATIFIED PREVALENCE
{'='*55}

This report shows RAW PREVALENCE DATA ONLY - no statistical
models or fitted lines. Each point represents the actual
percentage of people with each condition in that age bin.

SAMPLE SIZE
• Female: 12,313 subjects (52%)
• Male: 11,152 subjects (48%)
• Age range: 40-70 years
• Age bins: 40-45, 45-50, 50-55, 55-60, 60-65, 65-70

WHAT YOU SEE IN THE PLOTS
• Blue circles = Male prevalence at each age bin
• Red squares = Female prevalence at each age bin
• Error bars = 95% confidence intervals (Wilson score)
• NO fitted lines or statistical models

TOP CONDITIONS - MALES (Largest Increase 40→70)
"""
    for _, row in male_top.iterrows():
        summary += f"\n  • {row['condition']}: {row['male_prev_40']:.1f}% → {row['male_prev_70']:.1f}%"
    
    summary += "\n\nTOP CONDITIONS - FEMALES (Largest Increase 40→70)\n"
    for _, row in female_top.iterrows():
        summary += f"\n  • {row['condition']}: {row['female_prev_40']:.1f}% → {row['female_prev_70']:.1f}%"

    summary += """


KEY OBSERVATIONS

1. CARDIOVASCULAR CONDITIONS
   Hypertension, Hyperlipidemia show clear increases with age
   in BOTH genders, with males often having higher prevalence.

2. METABOLIC CONDITIONS
   Diabetes, Prediabetes increase with age in both genders.

3. MENTAL HEALTH
   Depression, Anxiety, ADHD show DECREASES with age - likely
   cohort effects (younger generations diagnosed more often).

4. SEX-SPECIFIC CONDITIONS
   Some conditions only appear in one gender (e.g., PCOS).
"""
    add_text_page(pdf, summary, fontsize=11)
    print("  Added: Executive summary")

    # ===== SUMMARY COMPARISON PLOT =====
    if add_image_page(pdf, os.path.join(OUT, 'summary_simple_prevalence.png'),
                      'Top Conditions by Prevalence'):
        print("  Added: Summary prevalence plot")

    # ===== TOP CONDITIONS TABLE - MALES =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    male_data = results[results['male_n_positive'] >= 50].nlargest(15, 'male_change')[
        ['condition', 'male_n_positive', 'male_prev_40', 'male_prev_70', 'male_change']
    ].copy()
    male_data['male_prev_40'] = male_data['male_prev_40'].round(1)
    male_data['male_prev_70'] = male_data['male_prev_70'].round(1)
    male_data['male_change'] = male_data['male_change'].round(1)
    male_data['male_n_positive'] = male_data['male_n_positive'].astype(int)
    male_data.columns = ['Condition', 'N', 'Prev@40 (%)', 'Prev@70 (%)', 'Change (pp)']

    table = ax.table(cellText=male_data.values, colLabels=male_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(male_data.columns)):
        table[(0, j)].set_facecolor('#2980B9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 15 Conditions - MALES (Largest Increase with Age)', 
                 fontsize=14, fontweight='bold', color='#2980B9', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Male top conditions table")

    # ===== TOP CONDITIONS TABLE - FEMALES =====
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    female_data = results[results['female_n_positive'] >= 50].nlargest(15, 'female_change')[
        ['condition', 'female_n_positive', 'female_prev_40', 'female_prev_70', 'female_change']
    ].copy()
    female_data['female_prev_40'] = female_data['female_prev_40'].round(1)
    female_data['female_prev_70'] = female_data['female_prev_70'].round(1)
    female_data['female_change'] = female_data['female_change'].round(1)
    female_data['female_n_positive'] = female_data['female_n_positive'].astype(int)
    female_data.columns = ['Condition', 'N', 'Prev@40 (%)', 'Prev@70 (%)', 'Change (pp)']

    table = ax.table(cellText=female_data.values, colLabels=female_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(female_data.columns)):
        table[(0, j)].set_facecolor('#E74C3C')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax.set_title('Top 15 Conditions - FEMALES (Largest Increase with Age)', 
                 fontsize=14, fontweight='bold', color='#E74C3C', pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print("  Added: Female top conditions table")

    # ===== INDIVIDUAL CONDITION PLOTS =====
    # Key conditions to show
    key_conditions = [
        'Hypertension', 'Diabetes', 'Ischemic_Heart_Disease', 'Osteoarthritis',
        'Depression', 'Anxiety', 'Allergy', 'ADHD',
        'Migraine', 'Gout', 'Anemia', 'Sleep_Apnea',
        'Glaucoma', 'Hearing_loss', 'Back_Pain', 'Obesity',
        'Hyperlipidemia', 'Prediabetes', 'Fatty_Liver_Disease',
        'Asthma', 'IBS', 'Psoriasis', 'Fractures'
    ]

    for cond in key_conditions:
        fpath = os.path.join(OUT, f'simple_prev_{cond}.png')
        if add_image_page(pdf, fpath):
            print(f"  Added: {cond}")

    # ===== METHODOLOGY PAGE =====
    methodology = """
METHODOLOGY
{'='*55}

DATA PRESENTATION
• RAW DATA ONLY - No statistical models or fitted lines
• Each point = actual prevalence in that age bin
• Age bins: 5-year intervals (40-45, 45-50, etc.)
• Error bars = 95% confidence intervals (Wilson score)

PLOT INTERPRETATION
• Blue circles = Male prevalence at each age bin
• Red squares = Female prevalence at each age bin
• Y-axis = Prevalence (%) - percentage with condition
• X-axis = Age midpoint of each 5-year bin

CONFIDENCE INTERVALS
• Wilson score interval for binomial proportions
• More accurate than normal approximation for small samples
• Asymmetric intervals that respect 0-100% bounds

INCLUSION CRITERIA
• Age: 40-70 years
• Minimum 50 cases per gender for analysis
• Total n ≥ 100 for condition

ADVANTAGES OF THIS APPROACH
• No model assumptions
• Shows actual data patterns
• Easy to interpret
• Confidence intervals show uncertainty

LIMITATIONS
1. No statistical testing of trends
2. Cannot extrapolate beyond observed ages
3. Some conditions are sex-specific
4. Cross-sectional design (not longitudinal)
5. Survival bias may affect older ages

SAMPLE SIZES
• Total: 23,473 subjects
• Female: 12,313 (52%)
• Male: 11,152 (48%)
"""
    add_text_page(pdf, methodology, fontsize=10)
    print("  Added: Methodology page")

    # ===== KEY FINDINGS PAGE =====
    findings = """
KEY FINDINGS BY CONDITION CATEGORY
{'='*55}

CARDIOVASCULAR
• Hypertension: Clear increase with age in both genders
  Males have higher prevalence at all ages
• Ischemic Heart Disease: Increases with age, much higher
  in males

METABOLIC
• Hyperlipidemia: Dramatic increase with age in both
  Males start higher but females catch up
• Diabetes: Increases with age in both genders
• Prediabetes: Similar pattern to diabetes

MENTAL HEALTH
• Depression: DECREASES with age in both genders
  Likely cohort effect (younger people diagnosed more)
• Anxiety: Similar decreasing pattern
• ADHD: Strong cohort effect visible

MUSCULOSKELETAL
• Osteoarthritis: Increases with age, steeper in females
• Back Pain: High prevalence, increases slightly with age
• Gout: Much more common in males, increases with age

AUTOIMMUNE/INFLAMMATORY
• Allergy: DECREASES with age in both genders
  Another cohort effect
• Psoriasis: Relatively stable across ages

SENSORY
• Hearing loss: Increases with age, higher in males
• Glaucoma: Increases with age in both genders


NOTES ON INTERPRETATION
• These are RAW DATA - no statistical modeling
• Patterns visible to the eye are real data patterns
• Gender differences may reflect:
  - Biological differences
  - Healthcare-seeking behavior
  - Diagnostic patterns
  - Survival differences
• Always check sample sizes (n) in legend
"""
    add_text_page(pdf, findings, fontsize=10)
    print("  Added: Key findings page")

print(f"\nPDF saved: {pdf_path}")
