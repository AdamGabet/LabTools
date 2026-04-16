# Understanding the Subject Modality CSV

## What the CSV Contains

The file `subject_test_date_overlap.csv` maps each HPP 10K patient (by `registration_code`) to all the medical tests/modality data they have, with dates.

## CSV Structure

| Column | Meaning |
|--------|---------|
| `registration_code` | Subject ID (format: `10K_XXXXXXXXXX`) |
| `date_<modality>` | Dates when subject had this test, semicolon-separated (e.g., `2023-01-15;2023-06-20`) |
| `n_dates_<modality>` | How many dates with this test (0 = no data) |
| `path_hint_<modality>` | Example file paths for quick verification |
| `n_modalities` | Total number of modalities with data for this subject |

## Modalities Available

| Modality | Source Type | Description |
|----------|-------------|-------------|
| `cgm` | signal | Continuous glucose monitor |
| `sleep` | signal | Sleep data (Itamar) |
| `dexa` | signal | Bone density / body composition |
| `ecg` | signal | Electrocardiogram |
| `retina` | signal | Retinal imaging |
| `ultrasound` | signal | Ultrasound images |
| `voice` | signal | Voice recordings |
| `abi` | signal | Ankle-brachial index |
| `gait` | signal | Gait analysis |
| `metabolites` | tabular | Untargeted metabolomics |
| `blood_test` | tabular | Blood panel results |
| `microbiome` | tabular | Gut microbiome |
| `nightingale` | tabular | Nightingale blood markers |
| `mental` | tabular | Mental health assessments |
| `proteomics` | tabular | Olink proteomics |

## Where to Find the Pre-built CSV

The CSV is already built and available at:
```
/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv
```

## How to Use It

### 1. Check which modalities a patient has
```python
import pandas as pd

df = pd.read_csv('research/multimodal_baseline/subject_test_date_overlap.csv')
subject_row = df[df['registration_code'] == '10K_1234567890'].iloc[0]

# List all modalities with data for this patient
modalities_with_data = []
for col in df.columns:
    if col.startswith('n_dates_') and subject_row[col] > 0:
        modality = col.replace('n_dates_', '')
        modalities_with_data.append(modality)
        
print(f"Patient has: {modalities_with_data}")
```

### 2. Find patients with specific modality on a specific date
```python
# Find patients who had CGM on 2023-01-15
cgm_col = 'date_cgm'
patients_with_cgm = df[df[cgm_col].str.contains('2023-01-15', na=False)]
print(patients_with_cgm['registration_code'].tolist())
```

### 3. Find patients with multiple modalities (multimodal)
```python
# Patients with at least 3 modalities
multimodal = df[df['n_modalities'] >= 3]
print(f"Found {len(multimodal)} multimodal patients")
```

### 4. Longitudinal subjects (repeated measures)
```python
# Patients with >1 CGM date
repeated_cgm = df[df['n_dates_cgm'] > 1]
print(f"Found {len(repeated_cgm)} patients with repeated CGM")
```

### 5. Find overlap between two modalities
```python
# Patients with both CGM and Sleep data
has_cgm = df['n_dates_cgm'] > 0
has_sleep = df['n_dates_sleep'] > 0
overlap = df[has_cgm & has_sleep]
print(f"Found {len(overlap)} patients with both CGM and Sleep")
```

## Example Output

```
registration_code,date_cgm,n_dates_cgm,date_sleep,n_dates_sleep,n_modalities
10K_1234567890,2023-01-15;2023-06-20,2,2023-01-15,1,3
10K_9876543210,,0,2023-02-10,1,1
```

This shows:
- Patient `10K_1234567890`: 2 CGM dates, 1 sleep date, 3 total modalities
- Patient `10K_9876543210`: no CGM, 1 sleep date, only 1 modality
