"""
Find metabolite KPIs in the 'sweet spot' r=0.50-0.75.
Too high (>0.75): trivially easy (carry-forward model wins).
Too low (<0.40): basically unpredictable noise.
Target: r=0.50-0.75 with clinical meaning and cross-modal relevance.
"""
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline'

stab_df = pd.read_csv(f'{OUTDIR}/metabolite_stability.csv')

sweet = stab_df[(stab_df['r_stability'] >= 0.45) & (stab_df['r_stability'] <= 0.75)]
print(f'Metabolites in r=0.45-0.75 range: {len(sweet)}')
print(sweet[['metabolite', 'r_stability', 'n_v2']].to_string(index=False))

# Clinically meaningful terms to look for in the sweet spot
clinical_terms = [
    'urate', 'uric', 'creatinine', 'glucose', 'insulin',
    'cortisol', 'testosterone', 'estrogen', 'estradiol', 'progesterone',
    'thyroxine', 'thyroid', 'bile', 'bilirubin', 'cholesterol',
    'carnitine', 'choline', 'betaine', 'homocysteine',
    'glutamine', 'glutamate', 'glycine', 'serine', 'alanine',
    'leucine', 'valine', 'isoleucine',  # BCAAs
    'tryptophan', 'serotonin', 'kynurenine', 'indole',
    'sphingosine', 'ceramide', 'lysophospho',
    'DHEA', 'androstenedione', 'androsterone',
    'vitamin', 'tocopherol', 'retinol',
    'lactate', 'pyruvate', 'citrate', 'succinate',
    'fibrinopeptide', 'peptide',
    'biliverdin', 'heme', 'porphyrin',
]

print('\n=== Clinically identifiable metabolites in sweet spot ===')
for _, row in sweet.iterrows():
    name = row['metabolite'].lower()
    for term in clinical_terms:
        if term.lower() in name:
            print(f"  r={row['r_stability']:.3f}  {row['metabolite']}")
            break

# Also show all metabolites in range, sorted by r descending
print('\n=== Full sweet spot list (r=0.50-0.75), most stable first ===')
sweet_sorted = sweet[(sweet['r_stability'] >= 0.50)].sort_values('r_stability', ascending=False)
print(sweet_sorted[['metabolite', 'r_stability']].to_string(index=False))
