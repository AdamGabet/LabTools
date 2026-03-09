# Secure Analysis Architecture

## Overview

This architecture enables Claude to perform meaningful data analysis while preventing access to raw data rows. The key principle: **Claude operates on cohort references, never sees raw data beyond approved samples.**

---

## Architecture Diagram

```
                              ┌─────────────────────────┐
                              │     USER REQUEST        │
                              └───────────┬─────────────┘
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               MAIN AGENT                                    │
│                      (No arbitrary code execution)                          │
│                                                                             │
│  Hooks enforced on ALL agents:                                              │
│  • Read: blocked on .workspace/, data/, *.parquet, *.csv, *.pkl             │
│  • Bash: blocked or restricted to safe commands                             │
│  • Grep: blocked on data directories                                        │
└─────────────────────────────────────────────────────────────────────────────┘
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ RESEARCH PHASE  │         │ ANALYSIS PHASE  │         │  OUTPUT PHASE   │
│   (Subagents)   │         │    (Tools)      │         │  (Subagents)    │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│                 │         │                 │         │                 │
│ research-       │         │ DATA PREP:      │         │ figure-creator  │
│ hypothesis      │────────▶│ • create_cohort │         │ (Bash, Write)   │
│ (Web, Read docs)│         │ • filter_cohort │         │                 │
│                 │         │ • merge_cohort  │         │ report-creator  │
│                 │         │ • split_cohort  │         │ (Bash, Write)   │
│                 │         │ • match_cohorts │         │                 │
│                 │         │                 │         │                 │
│                 │         │ INSPECT:        │         │                 │
│                 │         │ • describe      │         │                 │
│                 │         │ • value_counts  │         │                 │
│                 │         │ • check_balance │         │                 │
│                 │         │                 │         │                 │
│                 │         │ ANALYZE:        │         │                 │
│                 │         │ • correlation   │────────▶│                 │
│                 │         │ • regression    │ results │                 │
│                 │         │ • logistic      │ (inline)│                 │
│                 │         │ • compare_groups│         │                 │
│                 │         │ • association   │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │   SKEPTICAL REVIEWER    │
                      │   (WebSearch, WebFetch) │
                      │                         │
                      │   Must PASS before      │
                      │   proceeding to output  │
                      │                         │
                      │   Verdict: PASS/REVISE  │
                      └─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WORKSPACE (.workspace/)                            │
│                                                                             │
│   cohort_*.parquet     - Cohort data files (BLOCKED from all agents)        │
│   manifest.json        - Tracks cohorts + lineage (tools read internally)   │
│   results/*.json       - Analysis results                                   │
│                                                                             │
│   ⚠️  Claude references cohorts by ID only - never reads files directly     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Data Preparation Tools

Each tool operates on cohort IDs. Returns metadata + new cohort_id. Writes data to workspace.

### create_cohort
```python
create_cohort(
    systems: list[str],           # e.g., ['blood_lipids', 'glycemic_status']
    research_stage: str = None,   # filter to specific stage
    cohort_type: str = None       # e.g., '10K', 'BRCA', etc.
)
# Returns:
{
    "cohort_id": "abc123",
    "n_subjects": 10452,
    "n_rows": 15234,  # if multiple research stages
    "columns": ["cholesterol", "ldl", "hdl", "hba1c", ...],
    "systems_included": ["blood_lipids", "glycemic_status"],
    "index": ["RegistrationCode", "research_stage"]
}
```

### filter_cohort
```python
filter_cohort(
    cohort_id: str,
    age: tuple = None,            # e.g., (40, 70)
    gender: str = None,           # 'male', 'female'
    bmi: tuple = None,            # e.g., (18.5, 30)
    include_conditions: list = None,
    exclude_conditions: list = None,
    exclude_medications: list = None,
    require_columns: list = None,  # drop rows where these are missing
    custom_filter: str = None      # predefined safe filters only
)
# Returns:
{
    "cohort_id": "def456",
    "n_subjects": 3201,
    "n_rows": 4521,
    "filters_applied": ["age: 40-70", "gender: female", "excluded: diabetes"],
    "columns": [...],
    "parent_cohort": "abc123"
}
```

### merge_cohort
```python
merge_cohort(
    cohort_id: str,
    systems: list[str],           # additional systems to merge
    how: str = 'inner'            # 'inner', 'left', 'outer'
)
# Returns:
{
    "cohort_id": "ghi789",
    "n_subjects": 2845,
    "columns": [...],  # expanded with new system columns
    "merge_stats": {
        "original_n": 3201,
        "matched_n": 2845,
        "match_rate": "88.9%"
    }
}
```

### split_cohort
```python
split_cohort(
    cohort_id: str,
    by: str,                      # column to split on
    min_group_size: int = 100     # warn if groups are too small
)
# Returns:
{
    "splits": {
        "never_smoker": {"cohort_id": "split_a", "n": 1820},
        "former_smoker": {"cohort_id": "split_b", "n": 650},
        "current_smoker": {"cohort_id": "split_c", "n": 375}
    },
    "warnings": ["current_smoker group has n=375, consider pooling"]
}
```

### match_cohorts
```python
match_cohorts(
    cases_cohort: str,
    controls_cohort: str,
    match_on: list[str],          # e.g., ['age', 'gender', 'bmi']
    ratio: int = 1,               # controls per case
    caliper: float = 0.2          # matching tolerance
)
# Returns:
{
    "matched_cases_id": "matched_abc",
    "matched_controls_id": "matched_def",
    "n_cases": 500,
    "n_controls": 1000,
    "balance_check": {
        "age": {"smd": 0.02, "balanced": True},
        "gender": {"smd": 0.01, "balanced": True},
        "bmi": {"smd": 0.05, "balanced": True}
    }
}
```

### add_computed_column
```python
add_computed_column(
    cohort_id: str,
    name: str,
    formula: str                  # must be from approved formulas
)
# Approved formulas:
# - 'bin(column, [edges])'       → categorical bins
# - 'zscore(column)'             → standardized
# - 'log(column)'                → log transform
# - 'ratio(col_a, col_b)'        → division
# - 'sum([col_a, col_b, ...])'   → row sum
# - 'mean([col_a, col_b, ...])'  → row mean
# - 'flag_condition(condition)'  → binary from medical_conditions

# Returns:
{
    "cohort_id": "new123",
    "columns": [..., "new_column_name"],
    "new_column_stats": {"mean": 0.52, "std": 0.31, ...}
}
```

---

## Layer 2: Inspection Tools

### describe_cohort
```python
describe_cohort(
    cohort_id: str,
    show_sample: bool = True   # show 5 random rows from top 100
)
# Returns:
{
    "cohort_id": "abc123",
    "n_subjects": 2845,
    "n_rows": 3102,
    "columns": ["col1", "col2", ...],
    "dtypes": {"col1": "float64", "col2": "category", ...},
    "missing": {"col1": "2.3%", "col2": "0.0%", ...},
    "numeric_summary": {
        "age": {"mean": 54.2, "std": 8.1, "min": 40, "median": 53, "max": 70},
        "cholesterol": {"mean": 205, "std": 38, "min": 120, "median": 201, "max": 340}
    },
    "categorical_summary": {
        "gender": {"male": 1342, "female": 1503},
        "smoking": {"never": 1820, "former": 650, "current": 375}
    },
    "lineage": ["created from blood_lipids + glycemic", "filtered age 40-70"],

    # Only included when show_sample=True (default):
    "sample": [
        {"col1": 5.2, "col2": "male", ...},  # 5 random rows
        {"col1": 4.8, "col2": "female", ...},
        ...
    ]
}
# Note: sample is 5 random rows from top 100 rows (APPROVED DATA EXPOSURE)
```

### value_counts
```python
value_counts(cohort_id: str, column: str, top_n: int = 20)
# Returns:
{
    "column": "diagnosis",
    "counts": {
        "healthy": 4521,
        "hypertension": 1232,
        "diabetes": 890,
        ...
    },
    "n_unique": 45,
    "missing": 23
}
```

### check_balance
```python
check_balance(
    cohort_id: str,
    across: str,                  # grouping variable
    variables: list[str]          # variables to check balance
)
# Returns:
{
    "groups": ["treatment", "control"],
    "balance": {
        "age": {"group_means": [52.1, 52.4], "smd": 0.03, "balanced": True},
        "gender_female": {"group_props": [0.48, 0.51], "smd": 0.06, "balanced": True},
        "bmi": {"group_means": [27.2, 28.1], "smd": 0.15, "balanced": False}
    },
    "overall_balanced": False,
    "recommendation": "Consider adjusting for BMI or re-matching"
}
```

### list_cohorts
```python
list_cohorts()
# Returns:
[
    {
        "cohort_id": "abc123",
        "created": "2024-01-15 10:30:00",
        "n_subjects": 10452,
        "description": "blood_lipids + glycemic_status",
        "parent": None
    },
    {
        "cohort_id": "def456",
        "created": "2024-01-15 10:31:00",
        "n_subjects": 3201,
        "description": "filtered: age 40-70, female, non-diabetic",
        "parent": "abc123"
    }
]
```

---

## Layer 3: Analysis Tools

All analysis tools operate on cohort_ids and return only aggregated results.

### correlation
```python
correlation(
    cohort_id: str,
    col_a: str,
    col_b: str,
    method: str = 'pearson'       # 'pearson', 'spearman', 'kendall'
)
# Returns:
{
    "r": 0.342,
    "p_value": 0.0001,
    "n": 2845,
    "ci_95": [0.28, 0.40],
    "method": "pearson"
}
```

### correlation_matrix
```python
correlation_matrix(
    cohort_id: str,
    columns: list[str]
)
# Returns: dict of column pairs → correlation stats
```

### regression
```python
regression(
    cohort_id: str,
    features: list[str],
    target: str,
    model: str = 'ridge',         # 'ols', 'ridge', 'lasso', 'elastic_net'
    cv_folds: int = 5,
    scale_features: bool = True
)
# Returns:
{
    "model": "ridge",
    "metrics": {
        "r2": 0.342,
        "r2_cv": 0.318,
        "rmse": 0.82,
        "mae": 0.65
    },
    "coefficients": {
        "age": {"coef": 0.023, "se": 0.005, "p": 0.001},
        "bmi": {"coef": 0.156, "se": 0.021, "p": 0.0001},
        ...
    },
    "feature_importance": {
        "bmi": 0.32,
        "age": 0.18,
        ...
    },
    "n_samples": 2845,
    "n_features": 10
}
```

### logistic_regression
```python
logistic_regression(
    cohort_id: str,
    features: list[str],
    target: str,                  # binary outcome
    cv_folds: int = 5
)
# Returns:
{
    "metrics": {
        "auc": 0.78,
        "auc_cv": 0.75,
        "accuracy": 0.72,
        "precision": 0.68,
        "recall": 0.71
    },
    "odds_ratios": {
        "smoking": {"or": 2.3, "ci": [1.8, 2.9], "p": 0.0001},
        ...
    },
    "confusion_matrix": {
        "tp": 234, "fp": 89, "tn": 1821, "fn": 102
    }
}
```

### compare_groups
```python
compare_groups(
    cohort_id: str,
    group_col: str,
    outcomes: list[str],
    adjust_for: list[str] = None  # confounders
)
# Returns:
{
    "groups": ["never_smoker", "current_smoker"],
    "comparisons": {
        "cholesterol": {
            "means": {"never_smoker": 198, "current_smoker": 215},
            "raw_diff": 17,
            "adjusted_diff": 12,
            "p_value": 0.003,
            "effect_size": 0.35
        },
        ...
    }
}
```

### association_study
```python
association_study(
    cohort_id: str,
    exposure: str,
    outcome: str,
    confounders: list[str],
    method: str = 'auto'          # 'linear', 'logistic', 'auto'
)
# Returns:
{
    "exposure": "sleep_hours",
    "outcome": "hba1c",
    "method": "linear",
    "unadjusted": {
        "beta": -0.08,
        "se": 0.02,
        "p": 0.001,
        "ci": [-0.12, -0.04]
    },
    "adjusted": {
        "beta": -0.05,
        "se": 0.02,
        "p": 0.02,
        "ci": [-0.09, -0.01],
        "confounders_adjusted": ["age", "gender", "bmi"]
    },
    "n": 4521,
    "interpretation": "Each additional hour of sleep associated with 0.05 lower HbA1c"
}
```

---

## Layer 4: Subagents

### Research Hypothesis Subagent

Researches scientific literature and refines user requests into testable hypotheses that match available data.

**Tools:**
- `WebSearch` - search scientific literature
- `WebFetch` - read papers/abstracts
- `Read` - read body system documentation (column lists, not data files)
- NO data access, NO Bash

**Input:** User's research question + available body systems/columns
**Output:** Refined hypothesis, relevant literature, suggested analysis plan

**When to use:**
- User has a vague research question ("Is sleep related to heart disease?")
- Need to ground hypothesis in existing literature
- Need to map user's question to available variables in HPP data

```python
# Claude calls:
Task(
    subagent_type="research-hypothesis",
    prompt="""
    User question: "Is there a relationship between sleep and diabetes?"

    Available data:
    - Body systems: sleep, glycemic_status, Age_Gender_BMI, lifestyle
    - Sleep columns: sleep_hours, sleep_quality, ahi, odi, sleep_efficiency
    - Glycemic columns: hba1c, fasting_glucose, ogtt_2h, homa_ir

    Tasks:
    1. Search literature for known sleep-glycemic associations
    2. Identify specific testable hypotheses
    3. Recommend which variables to use as exposure/outcome
    4. Suggest confounders to adjust for
    5. Note any limitations of cross-sectional analysis
    """
)

# Returns:
{
    "literature_summary": "Multiple studies show short sleep (<6h) associated with...",
    "refined_hypotheses": [
        {
            "hypothesis": "Sleep duration is inversely associated with HbA1c",
            "exposure": "sleep_hours",
            "outcome": "hba1c",
            "confounders": ["age", "gender", "bmi", "physical_activity"],
            "expected_direction": "negative",
            "literature_support": "Cappuccio et al. 2010, Shan et al. 2015"
        },
        {
            "hypothesis": "Sleep apnea severity (AHI) is associated with insulin resistance",
            "exposure": "ahi",
            "outcome": "homa_ir",
            ...
        }
    ],
    "limitations": [
        "Cross-sectional design cannot establish causality",
        "Self-reported sleep duration may have measurement error"
    ],
    "suggested_analysis_plan": [
        "1. Filter to non-diabetic adults age 40-70",
        "2. Run association_study for each hypothesis",
        "3. Stratify by gender as sensitivity analysis"
    ]
}
```

### Skeptical Reviewer Subagent

A critical reviewer that evaluates analysis quality, methodology, and whether results are scientifically meaningful. **The main agent's goal is to pass this review.**

**Tools:**
- `WebSearch` - verify claims, check if methodology aligns with literature
- `WebFetch` - read methodological guidelines, effect size benchmarks
- NO data access, NO Bash, NO analysis tools
- Receives analysis summary as input only (cannot re-run analyses)

**Input:** Full analysis pipeline (cohort operations, methods used, results)
**Output:** Critical review with verdict (pass/fail/revise)

**Review criteria:**
- Statistical validity (appropriate tests, sample sizes, multiple comparisons)
- Confounding control (are the right confounders adjusted?)
- Effect size interpretation (statistically significant ≠ clinically meaningful)
- Data leakage risks (proper subject-level splits?)
- Selection bias (exclusion criteria reasonable?)
- Generalizability (HPP population limitations)

```python
# Claude calls after completing analysis:
Task(
    subagent_type="skeptical-reviewer",
    prompt="""
    Review this analysis for scientific rigor.

    Research question: "Is sleep duration associated with HbA1c?"

    Cohort pipeline:
    1. create_cohort(['sleep', 'glycemic_status', 'Age_Gender_BMI', 'lifestyle'])
    2. filter_cohort(age=(40,70), exclude_conditions=['diabetes'])
    3. filter_cohort(require_columns=['sleep_hours', 'hba1c'])
    Final n=5891

    Methods:
    - Linear regression with ridge regularization
    - Confounders: age, gender, bmi, physical_activity
    - 5-fold CV at subject level

    Results:
    - Adjusted beta: -0.05 (95% CI: -0.09, -0.01)
    - p-value: 0.02
    - R² = 0.12

    Interpretation: "Each additional hour of sleep associated with 0.05 lower HbA1c"
    """
)

# Returns:
{
    "verdict": "REVISE",  # PASS, REVISE, or FAIL
    "overall_assessment": "Methodology is sound but interpretation overstates findings",

    "statistical_review": {
        "status": "PASS",
        "notes": "Appropriate use of ridge regression, proper CV splitting"
    },

    "confounding_review": {
        "status": "CONCERN",
        "notes": "Missing key confounders: smoking, alcohol, sleep disorders (AHI). These are available in the data and should be included.",
        "recommendation": "Re-run with smoking_status, alcohol_consumption, ahi as additional confounders"
    },

    "effect_size_review": {
        "status": "CONCERN",
        "notes": "Beta of -0.05 HbA1c per hour of sleep is statistically significant but clinically marginal. HbA1c measurement error is ~0.1-0.2 units. A 2-hour sleep difference yields 0.1 HbA1c change - at the noise floor.",
        "recommendation": "Reframe as 'weak association' not 'associated with lower HbA1c'"
    },

    "bias_review": {
        "status": "PASS",
        "notes": "Exclusion of diabetics is appropriate for studying pre-disease associations. Age range 40-70 is well-powered in HPP."
    },

    "missing_analyses": [
        "Non-linear relationship check (sleep may have U-shaped association)",
        "Stratification by gender (known sex differences in sleep-metabolism)",
        "Sensitivity analysis excluding sleep apnea patients"
    ],

    "interpretation_review": {
        "status": "REVISE",
        "original": "Each additional hour of sleep associated with 0.05 lower HbA1c",
        "suggested": "A weak but statistically significant inverse association was observed between sleep duration and HbA1c (β=-0.05, p=0.02), though the effect size is of uncertain clinical significance."
    },

    "required_changes_to_pass": [
        "Add smoking, alcohol, AHI as confounders",
        "Test for non-linear relationship",
        "Revise interpretation to reflect modest effect size"
    ]
}
```

**Main agent workflow:**
1. Complete analysis
2. Submit to skeptical-reviewer
3. If REVISE/FAIL: address required changes
4. Re-submit until PASS
5. Only then proceed to figures and report

### Figure Creator Subagent

Claude passes aggregated results to this subagent, which creates visualizations.

**Tools:**
- `Bash` - run Python/matplotlib scripts
- `Write` - save figure files and plotting scripts
- `Read` - ONLY allowed paths: `research/`, `figures/`, `.claude/templates/`
- Same hooks apply (blocked from .workspace/, data/, *.parquet, *.csv)

**Security: Data passed inline, not by file reference**
```python
# CORRECT - data in prompt:
Task(prompt="Create plot. Data: {'x': [1,2,3], 'y': [4,5,6]}")

# WRONG - file reference (blocked by hooks anyway):
Task(prompt="Create plot from .workspace/cohort_abc.parquet")
```

**Input:** Aggregated data (never raw rows) passed directly in prompt
**Output:** Path to saved figure

```python
# Claude calls:
Task(
    subagent_type="figure-creator",
    prompt="""
    Create a forest plot.

    Data:
    {
        "variables": ["Age", "BMI", "Smoking", "Exercise"],
        "odds_ratios": [1.02, 1.45, 2.30, 0.75],
        "ci_low": [0.98, 1.22, 1.80, 0.60],
        "ci_high": [1.06, 1.72, 2.95, 0.94],
        "p_values": [0.32, 0.001, 0.0001, 0.02]
    }

    Settings:
    - Title: "Risk Factors for Diabetes"
    - Reference line at OR=1
    - Color significant (p<0.05) in red
    - Save to: figures/diabetes_risk_factors.png
    """
)
```

### Report Creator Subagent

Compiles findings and figures into publication-ready report.

**Tools:**
- `Bash` - run report generation (LaTeX, markdown→PDF)
- `Write` - save report files
- `Read` - ONLY allowed paths: `research/`, `figures/`, `.claude/templates/`
- Same hooks apply (blocked from .workspace/, data/, *.parquet, *.csv)

**Security: Findings passed inline, figures by path (images only)**
```python
# CORRECT:
Task(prompt="""
  Create report.
  Findings: {"beta": -0.05, "p": 0.02, ...}
  Figures: ["research/study/fig1.png", "research/study/fig2.png"]
""")
```

**Input:** Findings dict (inline), figure paths (images only)
**Output:** PDF report path

```python
Task(
    subagent_type="report-creator",
    prompt="""
    Create a research report.

    Title: "Sleep Duration and Glycemic Control"

    Findings:
    {
        "main_result": {...},
        "sensitivity_analyses": {...},
        "subgroup_analyses": {...}
    }

    Figures:
    - figures/main_finding.png
    - figures/subgroups.png

    Output: research/sleep_glycemic/report.pdf
    """
)
```

---

## Workspace Management

### File Structure
```
.workspace/
├── manifest.json              # tracks all cohorts
├── cohort_abc123.parquet      # actual data (Claude can't read)
├── cohort_def456.parquet
└── results/
    ├── analysis_001.json      # analysis results
    └── analysis_002.json
```

### Manifest Schema
```json
{
    "cohorts": {
        "abc123": {
            "created": "2024-01-15T10:30:00",
            "n_subjects": 10452,
            "n_rows": 15234,
            "columns": ["col1", "col2"],
            "parent": null,
            "operations": ["create_cohort(['blood_lipids', 'glycemic'])"],
            "file": "cohort_abc123.parquet"
        }
    }
}
```

### Cleanup Tools
```python
delete_cohort(cohort_id)       # remove cohort file
cleanup_workspace(keep_last=5) # keep only recent cohorts
describe_lineage(cohort_id)    # show full derivation history
```

---

## Security Implementation

### Hooks Configuration

```json
// .claude/settings.local.json
{
    "hooks": {
        "Bash": {
            "pre": "python hooks/block_arbitrary_code.py"
        },
        "Read": {
            "pre": "python hooks/block_data_files.py"
        },
        "Grep": {
            "pre": "python hooks/block_data_dirs.py"
        }
    }
}
```

### Hook: block_data_files.py
```python
import sys
import json
import os

# Block by extension
BLOCKED_EXTENSIONS = ['.csv', '.parquet', '.pkl', '.feather', '.h5']

# Block these paths entirely
BLOCKED_PATHS = ['.workspace/', 'data/', '/mnt/data/']

# Allowlist for safe directories (code, docs, figures, research outputs)
ALLOWED_PATHS = [
    '.claude/',
    'research/',
    'figures/',
    'body_system_loader/',
    'predict_and_eval/',
    # Add other code directories
]

request = json.load(sys.stdin)
file_path = request.get('file_path', '')
file_path_normalized = os.path.normpath(file_path)

# Block data file extensions
if any(file_path.endswith(ext) for ext in BLOCKED_EXTENSIONS):
    print(f"BLOCKED: Cannot read data file {file_path}", file=sys.stderr)
    sys.exit(1)

# Block protected data paths
if any(blocked in file_path for blocked in BLOCKED_PATHS):
    print(f"BLOCKED: Cannot read from protected path {file_path}", file=sys.stderr)
    sys.exit(1)

# Optional: strict allowlist mode (uncomment to enable)
# if not any(allowed in file_path for allowed in ALLOWED_PATHS):
#     print(f"BLOCKED: Path not in allowlist {file_path}", file=sys.stderr)
#     sys.exit(1)
```

### Hook: block_arbitrary_code.py
```python
import sys

command = sys.stdin.read()

# Only allow specific commands
ALLOWED_PATTERNS = [
    'python -m secure_api',      # our tool runner
    'python secure_api/',        # our tool modules
    # Add other safe commands
]

if not any(pattern in command for pattern in ALLOWED_PATTERNS):
    print(f"BLOCKED: Arbitrary code execution not allowed", file=sys.stderr)
    sys.exit(1)
```

---

## Folder Structure

```
LabTools/
├── body_system_loader/          # existing - data loading
├── predict_and_eval/            # existing - ML utilities
├── secure_api/                  # NEW - privacy-preserving tools
│   ├── __init__.py
│   ├── workspace.py             # Workspace class, manifest management
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_prep.py         # create_cohort, filter, merge, split, match
│   │   ├── inspect.py           # describe_cohort, value_counts, check_balance
│   │   └── analyze.py           # correlation, regression, association_study
│   └── hooks/
│       ├── __init__.py
│       ├── block_data_files.py  # Read hook
│       └── block_arbitrary_code.py  # Bash hook
├── .claude/
│   ├── secure-analysis-architecture.md  # this doc
│   └── (subagent prompts)
└── research/                    # analysis outputs
    └── <study_name>/
        ├── analysis.py
        ├── report.pdf
        └── figures/
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `.workspace/` directory management
- [ ] Implement manifest.json tracking
- [ ] Set up hooks for blocking data access

### Phase 2: Data Preparation Tools
- [ ] `create_cohort()` - load and combine body systems
- [ ] `filter_cohort()` - apply filters
- [ ] `merge_cohort()` - add more systems
- [ ] `split_cohort()` - create subgroups
- [ ] `match_cohorts()` - case-control matching
- [ ] `add_computed_column()` - safe derived variables

### Phase 3: Inspection Tools
- [ ] `describe_cohort(show_sample=True)` - full metadata + 5 random rows from top 100
- [ ] `value_counts()` - categorical distributions
- [ ] `check_balance()` - covariate balance
- [ ] `list_cohorts()` - workspace overview

### Phase 4: Analysis Tools
- [ ] `correlation()` / `correlation_matrix()`
- [ ] `regression()` - linear models
- [ ] `logistic_regression()` - binary outcomes
- [ ] `compare_groups()` - group comparisons
- [ ] `association_study()` - exposure-outcome with confounders

### Phase 5: Subagents
- [ ] Research hypothesis subagent (literature search, refine questions)
- [ ] Skeptical reviewer subagent (critical review, must pass before reporting)
- [ ] Figure creator subagent prompt/instructions
- [ ] Report creator subagent prompt/instructions
- [ ] Integration with existing create-report skill

---

## Example Complete Workflow

```
User: "Investigate whether sleep duration is associated with HbA1c in non-diabetic adults, controlling for age, gender, BMI, and physical activity."

Claude:

1. create_cohort(systems=['sleep', 'glycemic_status', 'Age_Gender_BMI', 'lifestyle'])
   → cohort_id='c1', n=10452

2. filter_cohort('c1', age=(40,70), exclude_conditions=['diabetes', 'prediabetes'])
   → cohort_id='c2', n=6234

3. describe_cohort('c2')  # show_sample=True by default
   → Check distributions, missingness, plus see 5 random rows

4. filter_cohort('c2', require_columns=['sleep_hours', 'hba1c', 'physical_activity'])
   → cohort_id='c3', n=5891 (dropped missing)

5. association_study(
       cohort_id='c3',
       exposure='sleep_hours',
       outcome='hba1c',
       confounders=['age', 'gender', 'bmi', 'physical_activity']
   )
   → {adjusted_beta: -0.05, p: 0.02, ...}

6. Task(skeptical-reviewer, analysis_pipeline=...)
   → verdict: REVISE, missing confounders (smoking, AHI)

7. Re-run with additional confounders, add sensitivity analyses
   → Updated results

8. Task(skeptical-reviewer, analysis_pipeline=...)
   → verdict: PASS

9. split_cohort('c3', by='gender')
   → subgroup cohorts for sensitivity analysis

10. Task(figure-creator, data=results, type='forest_plot')
    → figures/sleep_hba1c_association.png

11. Task(report-creator, findings=all_results)
    → research/sleep_glycemic/report.pdf
```

---

## Open Questions

1. **Formula safety**: How to validate computed column formulas are safe?
2. **Custom analyses**: Process for adding new analysis tools when needed?
3. **Subagent implementation**: MCP tools vs Task tool for subagents?
4. **Performance**: Caching strategy for large cohort operations?
5. **Audit logging**: What level of detail for compliance?
