"""Configuration dataclass for BuildResults pipeline."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union


def _load_lab_config() -> dict:
    """Read lab_config.env from the same directory as this file."""
    config_path = Path(__file__).parent / 'lab_config.env'
    paths = {}
    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    paths[key.strip()] = val.strip()
    return paths


_LAB_PATHS = _load_lab_config()

# Base paths — override via environment variable or edit lab_config.env
JAFAR_ROOT = os.getenv('JAFAR_ROOT', _LAB_PATHS.get('JAFAR_ROOT', '/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam'))
GENIE_ROOT = os.getenv('GENIE_ROOT', _LAB_PATHS.get('GENIE_ROOT', '/net/mraid20/export/genie/LabData'))
SEGAL_GENIE_ROOT = os.getenv('SEGAL_GENIE_ROOT', _LAB_PATHS.get('SEGAL_GENIE_ROOT', '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData'))


@dataclass
class BuildResultsConfig:
    """
    Configuration for running cross-validation on multiple body systems.
    
    run_list format options:
        - 'system_name'                          # existing body system
        - {'name': '/path/to/file.csv'}          # csv file
        - {'name': ['col1', 'col2']}             # columns from other systems
    
    target_systems: same format as run_list
    
    column_descriptions format:
        {'column_name': 'regression' | 'classification' | 'ordinal'}
    """
    # Feature sources to evaluate
    run_list: List[Union[str, Dict]] = field(default_factory=list)
    
    # Target systems to predict
    target_systems: List[Union[str, Dict]] = field(default_factory=list)
    
    # Baseline model for comparison (same format as run_list items)
    baseline: Optional[Union[str, Dict]] = None
    
    # Confounders to regress out from features
    confounders: List[str] = field(default_factory=list)
    
    # Column type overrides for run_list columns
    run_column_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Column type overrides for target columns  
    target_column_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # Output directory
    save_dir: str = ''
    
    # Cross-validation settings
    num_seeds: int = 10
    num_splits: int = 5
    stratified_minority_threshold: float = 0.2  # Use stratified if minority < 20%
    
    # Execution settings
    num_threads: int = 8
    merge_closest_research_stage: bool = True
    testing: bool = False
    with_queue: bool = False
    resume_seeds: bool = False  # If True, load existing results and run only missing seeds
    only_labels: List[str] = None  # If set, only run these specific label names (skip others)
    ordinal_as_regression: bool = True  # If True, treat ordinal targets as regression (faster)

    # Ensemble settings (run after the main CV loop)
    ensemble_after_run: bool = False  # If True, run ensemble after all predictions are saved
    ensemble_systems: List[str] = None  # Feature systems to include; None = all
    ensemble_skip_systems: List[str] = None  # Feature systems to exclude (e.g. ['baseline'])


# Default configuration for gait prediction experiments
DEFAULT_CONFIG = BuildResultsConfig(
    run_list=[
        # Full body baselines
        {'single_cycle': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token/variant1_mean_max/test_eval_median_pooled.csv"},
        {'long_seq': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        
        # 8ze9xt8e_cls_token_masked - per body part (single cycle)
        {'single_masked_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked/arms/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_masked_head': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked/head/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_masked_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked/legs/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_masked_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked/torso/variant1_mean_max/test_eval_median_pooled.csv"},
        
        # lqc2exvo_long_all_masked - per body part (long seq)
        {'long_masked_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/arms/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_masked_head': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/head/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_masked_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/legs/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_masked_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/torso/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        
        # 8ze9xt8e_cls_token_masked_but_one - all but one body part (single cycle)
        {'single_all_but_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked_but_one/all_but_arms_masked/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_all_but_head': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked_but_one/all_but_head_masked/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_all_but_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked_but_one/all_but_legs_masked/variant1_mean_max/test_eval_median_pooled.csv"},
        {'single_all_but_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/8ze9xt8e_cls_token_masked_but_one/all_but_torso_masked/variant1_mean_max/test_eval_median_pooled.csv"},
        
        # lqc2exvo_long_all_masked_but_one - all but one body part (long seq)
        {'long_all_but_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_arms_masked/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_all_but_head': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_head_masked/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_all_but_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_legs_masked/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
        {'long_all_but_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_torso_masked/variant5_merged_groups_percentile/test_eval_median_pooled.csv"},
    ],
    target_systems=[
        # Core baselines
        {'age_gender_bmi_VAT': ['age', 'gender', 'bmi', 'total_scan_vat_area']},
        {"anthropometric_group": ["waist", "hips", "weight"]},
        
        # Bone density - spine measurements (very strong effects)
        {'bone_density_m': [
            'spine_l1_l3_area', 'spine_l2_l4_area', 'spine_l1_l4_area',
            'spine_l1_l3_bmc', 'spine_l2_l4_bmc', 'spine_l1_l4_bmc',
            'body_spine_area', 'spine_l1_l2_bmd'
        ]},
        
        # Body composition - height and mass distribution
        {'body_composition_m': [
            'height', 'body_comp_arm_right_bone_mass', 'total_scan_vat_area',
            'body_comp_arm_left_lean_mass', 'body_comp_arms_lean_mass',
            'body_comp_arm_left_fat_free_mass', 'body_comp_arms_fat_free_mass',
            'body_comp_arm_right_lean_mass'
        ]},
        
        # Cardiovascular - ECG and vascular
        {'cardiovascular_system_m': [
            'r_mv_V3', 'r_mv_aVR', 'from_l_thigh_to_l_ankle_duration',
            'from_r_thigh_to_r_ankle_duration', 's_mv_V2', 's_mv_V3',
            'automorph_fractal_dimension', 'st_mv_V1'
        ]},
        
        # Liver function
        {'liver_m': [
            'bt__alt_gpt', 'liver_elasticity', 'bt__alkaline_phosphatase',
            'bt__ast_got', 'liver_viscosity', 'liver_sound_speed'
        ]},
        
        # Sleep metrics
        {'sleep_group_m': [
            'rem_latency', 'sleep_efficiency', 'heart_rate_min_during_sleep',
            'rdi', 'heart_rate_mean_during_sleep', 'csr_percent',
            'total_sleep_time', 'saturation_mean'
        ]},
        
        # Blood markers
        {'hematopoietic_m': [
            'bt__mchc', 'bt__ferritin', 'bt__mcv', 'bt__hct'
        ]},
        
        # Top proteomics markers
        {'proteomics_m': [
            'CTSH', 'CPE', 'SAV1', 'CD79B', 'DPP10', 'PNLIPRP1', 'COMT', 'ODAM'
        ]},
        
        # Nightingale metabolomics
        {'nightingale_m': [
            'GlycA', 'XL_HDL_CE', 'XL_HDL_C', 'VLDL_size',
            'XL_HDL_L', 'L_HDL_CE', 'L_HDL_C', 'XL_HDL_PL'
        ]},
        
        # Immune markers
        {'immune_system_m': ['bt__neutrophils_%', 'bt__monocytes_abs']},
        
        # Glycemic status
        {'glycemic_status_m': ['iglu_1st_quartile', 'iglu_adrr']},
        
        # Frailty - grip strength and lean mass
        {'frailty_m': [
            'hand_grip_left', 'hand_grip_right',
            'frailty_body_comp_arm_right_lean_mass', 'frailty_body_comp_arm_left_lean_mass',
            'frailty_body_comp_leg_left_lean_mass', 'frailty_body_comp_leg_right_lean_mass'
        ]},
        
        # Lifestyle and activity
        {'lifestyle_group_m': [
            'high_exercise_duration_Between 2 and 3 hours', 'white_wine_glasses_week',
            'usual_walking_pace_Slow', 'vigorous_activity_minutes', 'moderate_activity_minutes',
            'sleep_hours_in_24H', 'walking_minutes_day', 'walking_10min_days_a_week'
        ]},
        
        # Lipids
        {'blood_tests_lipids_m': [
            'bt__non_hdl_cholesterol', 'bt__hdl_cholesterol',
            'bt__total_cholesterol', 'bt__triglycerides'
        ]},
        
        # Renal
        {'renal_function_m': ['bt__creatinine']},
        
    ],
    #baseline={'Age_Gender_BMI_VAT': ['age', 'gender', 'total_scan_vat_area', 'bmi']},
    confounders=["height", "gender", "bmi", "total_scan_vat_area", "age"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/making_eval_test/',
    with_queue=True,
)

# Default configuration for gait prediction experiments
ABLATION_ALL_CONFIG = BuildResultsConfig(
    run_list=[
        # Full body baselines
        {'long_seq': f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
        # lqc2exvo_long_all_masked - per body part (long seq)
        {'long_masked_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/arms/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_masked_head': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/head/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_masked_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/legs/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_masked_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked/torso/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        # lqc2exvo_long_all_masked_but_one - all but one body part (long seq)
        {'long_all_but_arms': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_arms_masked/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_all_but_head': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_head_masked/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_all_but_legs': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_legs_masked/variant5_merged_groups_percentile/cleaned_pooled.csv"},
        {'long_all_but_torso': f"{JAFAR_ROOT}/results/embeddings_11_30/lqc2exvo_long_all_masked_but_one/all_but_torso_masked/variant5_merged_groups_percentile/cleaned_pooled.csv"},
    ],
    target_systems=[
        # Core baselines
        {'age_gender_bmi_VAT': ['age', 'gender', 'bmi', 'total_scan_vat_area']},
        
        # Anthropometric group
        {"anthropometric_group": ["waist", "hips"]},
        
        # Blood lipids group
        {'blood_tests_lipids_group': [
            'bt__hdl_cholesterol', 'bt__non_hdl_cholesterol', 'bt__triglycerides'
        ]},
        
        # Body composition group - lean mass, tissue mass, and fat per body part
        {'body_composition_group': [
            'height',  # Overall height
            'waist',  # Anthropometric
            'total_scan_vat_area',  # Visceral fat
            'body_comp_arms_lean_mass',  # Arms lean mass
            'body_comp_arms_tissue_mass',  # Arms tissue mass
            'body_comp_arms_fat_mass',  # Arms fat mass
            'body_comp_legs_lean_mass',  # Legs lean mass
            'body_comp_legs_fat_mass',  # Legs fat mass
            'body_comp_trunk_lean_mass',  # Trunk lean mass
            'body_comp_trunk_fat_mass',  # Trunk fat mass
            'body_comp_android_fat_mass',  # Android region fat
            'body_comp_android_tissue_mass',  # Android region tissue
            'body_comp_gynoid_fat_mass',  # Gynoid region fat
            'body_comp_gynoid_tissue_mass',  # Gynoid region tissue
            'body_comp_total_lean_mass',  # Total lean mass
            'body_comp_total_fat_mass',  # Total body fat
            'body_comp_total_bone_mass',  # Total bone mass
        ]},
        
        # Bone density group - BMD and BMC per body part
        {'bone_density_group': [
            'spine_l1_l4_bmd',  # Spine BMD
            'spine_l1_l4_bmc',  # Spine BMC
            'body_head_bmd',  # Head BMD
            'body_head_bmc',  # Head BMC
            'body_arms_bmd',  # Arms BMD
            'body_arms_bmc',  # Arms BMC
            'body_legs_bmd',  # Legs BMD
            'body_legs_bmc',  # Legs BMC
            'body_trunk_bmd',  # Trunk BMD
            'body_trunk_bmc',  # Trunk BMC
            'body_pelvis_bmd',  # Pelvis BMD
            'body_pelvis_bmc',  # Pelvis BMC
            'femur_total_mean_bmd',  # Femur BMD
            'femur_total_mean_bmc',  # Femur BMC
            'body_total_bmd',  # Total body BMD
            'body_total_bmc',  # Total body BMC
        ]},
        
        # Cardiovascular system group
        {'cardiovascular_system_group': [
            'r_mv_V3', 'r_mv_V2', 'r_ms_V3',
            'sitting_blood_pressure_pulse_rate', 'q_ms_aVL',
            't_mv_I', 'hr_bpm', 'automorph_vein_fractal_dimension',
            'r_r_ms', 'automorph_fractal_dimension',
            'lying_blood_pressure_pulse_rate', 't_mv_V1',
            't_mv_V5', 'qt_ms', 't_mv_V4',
            'standing_three_min_blood_pressure_pulse_rate',
            'intima_media_th_1_fit', 'automorph_artery_fractal_dimension',
            'intima_media_th_2_fit', 'r_mv_V5', 'r_mv_V4', 'r_mv_I'
        ]},

        {'lifestyle_ordinal': f"{GENIE_ROOT}/Analyses/10K_Trajectories/body_systems/lifestyle_ordinal.csv"},
        {"exercise_label": f"{JAFAR_ROOT}/skeleton_data/subject_data/wearable_exercise_labels.csv"},
        
        # Frailty group
        {'frailty_group': [
            'hand_grip_left', 'hand_grip_right',
            'frailty_body_comp_arm_right_lean_mass',
            'frailty_body_comp_arm_left_lean_mass',
            'frailty_body_comp_leg_left_lean_mass',
            'frailty_body_comp_leg_right_lean_mass'
        ]},
        
        # Hematopoietic group
        {'hematopoietic_group': [
            'bt__mchc', 'bt__ferritin', 'bt__hct'
        ]},
        
        # Lifestyle group
        {'lifestyle_group': [
            'vigorous_activity_minutes', 'walking_10min_days_a_week',
            'physical_activity_maderate_days_a_week',
            'high_exercise_duration_Between 2 and 3 hours',
            'work_days_a_week', 'usual_walking_pace_Slow',
            'moderate_activity_minutes', 'sleep_hours_in_24H',
            'usual_walking_pace_Moderate',
            'high_exercise_duration_Between half an hour and an hour',
            'high_exercise_times_a_month_Two or three times a week',
            'usual_walking_pace_Fast', 'climb_staires_tymes_a_day_Zero',
            'usual_walking_pace_Average', 'walking_minutes_day'
        ]},
        
        # Liver group
        {'liver_group': [
            'liver_elasticity', 'bt__alkaline_phosphatase',
            'liver_viscosity', 'liver_sound_speed'
        ]},
        
        # Mental group
        {'mental_group': [
            'risks_taker', 'worry_long_after_embarrassment', 'worrier'
        ]},
        
        # Metabolites annotated group
        {'metabolites_annotated_group': [
            'LysoPE(P-16_0_0_0)',
            'Isopropyl 4-hydroxybenzoate_propyl 4-hydroxybenzoate_Propyl 2-furanacrylate',
            'LPE(14_0_0_0)',
            'xanthurenate _4 6-Dihydroxy-2-quinolinecarboxylic acid_Zeanic acid',
            'Cortolone-3-glucuronide', '4-Vinylphenol sulfate',
            'indolelactate_	3-Indolehydracrylic acid',
            'Sphingosine 1-phosphate (d16_1-P)'
        ]},
        
        # Nightingale group
        {'nightingale_group': [
            'XL_HDL_PL', 'L_HDL_FC', 'XL_HDL_C', 'XL_HDL_CE',
            'L_HDL_L', 'L_VLDL_C', 'GlycA', 'VLDL_size',
            'L_VLDL_FC', 'XL_HDL_L', 'L_HDL_CE', 'L_HDL_C',
            'HDL_size', 'XL_HDL_P', 'L_HDL_P', 'L_HDL_PL',
            'XL_VLDL_TG', 'XL_VLDL_P', 'L_VLDL_PL', 'XL_VLDL_L',
            'L_VLDL_TG'
        ]},
        
        # Sleep group
        {'sleep_group_group': [
            'heart_rate_min_during_sleep', 'rdi',
            'heart_rate_mean_during_sleep', 'saturation_mean'
        ]},
        
        # Subject group
        {'subject_group': ['age', 'bmi']},
        
        # Wearable monthly group
        {'wearable_monthly_group': [
            'wearable_weightlifting_monthly_hours',
            'wearable_walking_monthly_hours'
        ]},
        
    ],
    #baseline={'Age_Gender_BMI_VAT': ['age', 'gender', 'total_scan_vat_area', 'bmi']},
    confounders=["height", "gender", "bmi", "total_scan_vat_area", "age"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/masking_ablation_all/',
    with_queue=True,
)

MEDICAL_CONDITIONS_CONFIG = BuildResultsConfig(
    run_list = [
    # {JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant2_joint_stats/all_clean.csv
    {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
    "gait",
    {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"}
        #{"cycle_last_third": f"{JAFAR_ROOT}/results/best_embeddings/qjo95s37/all_last_third_embeddings.csv"}, 
    ],

    run_column_descriptions = {"activity": "classification", "seq_idx": "ordinal"},
    baseline = {'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},  # Use existing body system directly
    confounders = ['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    target_systems=[
        {'medical_conditions_grouped': f"{JAFAR_ROOT}/gil_link/Analyses/10K_Trajectories/body_systems/medical_conditions_grouped.csv"},
        "medical_conditions",
        "medicationsf"
    ],
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/medical_conditions_predictions/',
    with_queue=True,
)


# Mental group - new run with 15 seeds
MENTAL_CONFIG = BuildResultsConfig(
    run_list = [
    # {JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant2_joint_stats/all_clean.csv
    {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
    "gait",
    {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"}
        #{"cycle_last_third": f"{JAFAR_ROOT}/results/best_embeddings/qjo95s37/all_last_third_embeddings.csv"}, 
    ],
    target_systems=[
        'mental'
    ],
    baseline={'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},
    confounders=['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/long_vs_short_vs_features_vs_baseline_confounders2/',
    num_seeds=15,
    with_queue=True,
)


# Gait only - no baseline, no confounders
# Shows absolute predictive power of gait alone
GAIT_ONLY_CONFIG = BuildResultsConfig(
    run_list=[
        {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
        {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"},
    ],
    target_systems=[
        {"sleep_group": [
            'total_sleep_time', 'sleep_efficiency', 'sleep_latency', 'rem_latency',
            'total_rem_sleep_time', 'total_deep_sleep_time', 'total_light_sleep_time',
            'percent_of_rem_sleep_time', 'percent_of_deep_sleep_time', 'percent_of_light_sleep_time',
            'ahi', 'odi', 'rdi', 'csr_percent',
            'saturation_mean', 'saturation_min_value', 'hypoxic_burden',
            'heart_rate_mean_during_sleep', 'heart_rate_max_during_sleep', 'heart_rate_min_during_sleep'
        ]},
        'liver',
        'nightingale',
        {'lifestyle_group': [
            'sleep_hours_in_24H', 'hours_using_computer_not_work', 'hours_watching_tv', 'hours_driving',
            'work_hours_day', 'work_hours', 'work_days_a_week',
            'hours_outdoors_summer', 'hours_outdoors_winter',
            'walking_minutes_day', 'moderate_activity_minutes', 'vigorous_activity_minutes',
            'walking_10min_days_a_week', 'physical_activity_maderate_days_a_week',
            'manual_physical_work_Always', 'manual_physical_work_no', 'manual_physical_work_sometimes', 'manual_physical_work_usually',
            'climb_staires_tymes_a_day_1-5', 'climb_staires_tymes_a_day_6-10', 'climb_staires_tymes_a_day_11-15',
            'climb_staires_tymes_a_day_16-20', 'climb_staires_tymes_a_day_More than 20 times', 'climb_staires_tymes_a_day_Zero',
            'high_exercise_times_a_month_Every day', 'high_exercise_times_a_month_Four to five times a week',
            'high_exercise_times_a_month_Once a week', 'high_exercise_times_a_month_Two or three times a week',
            'high_exercise_times_a_month_Once in the last month', 'high_exercise_times_a_month_Two or three times in the last month',
            'high_exercise_duration_Less than a quarter of an hour', 'high_exercise_duration_Between a quarter of an hour and half an hour',
            'high_exercise_duration_Between half an hour and an hour', 'high_exercise_duration_Between an hour and an hour and a half',
            'high_exercise_duration_Between an hour and a half and two hours', 'high_exercise_duration_Between 2 and 3 hours',
            'high_exercise_duration_Over 3 hours',
            'usual_walking_pace_Slow', 'usual_walking_pace_Moderate', 'usual_walking_pace_Average', 'usual_walking_pace_Fast',
            'beer_cider_pints_week', 'white_wine_glasses_week', 'fortified_wine__glasses_week', 'liqueurs_measures_week', 'other_alcoholic_glasses_week'
        ]},
        'cardiovascular_system',
        'proteomics',
        'bone_density',
        'body_composition',
        'immune_system',
        'renal_function',
        'glycemic_status',
        'hematopoietic',
        'frailty',
        'blood_tests_lipids',
        "high_level_diet_with_stage", 
        "mental",
        "gait",
        "microbiome",
    ],
    baseline=None,  # No baseline - show absolute gait performance
    confounders=[],  # No confounders - raw gait predictions
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/gait_only_no_confounders/',
    num_seeds=15,
    with_queue=True,
)


# Labels needing more seeds (50 seeds, resume from existing 15)
# These are ensemble + long_seq labels with p < 0.1 that could become significant
# Uses only_labels to filter - results saved in same directory structure as original
ADDING_METABOLOMICS_CONFIG = BuildResultsConfig(
    run_list = [
    # {JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant2_joint_stats/all_clean.csv
    {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
    "gait",
    {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"}
        #{"cycle_last_third": f"{JAFAR_ROOT}/results/best_embeddings/qjo95s37/all_last_third_embeddings.csv"}, 
    ],
    target_systems=[
        'metabolites_unannotated',
    ],
    baseline={'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},
    confounders=['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/long_vs_short_vs_features_vs_baseline_confounders2/',
    num_seeds=15,
    resume_seeds=False,  # Force full redo - ran with wrong file
    with_queue=True,
    ordinal_as_regression=True,
)

ADDING_METABOLOMICS_GAIT_ONLY_CONFIG = BuildResultsConfig(
    run_list = [
    # {JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant2_joint_stats/all_clean.csv
    {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
    {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"}
        #{"cycle_last_third": f"{JAFAR_ROOT}/results/best_embeddings/qjo95s37/all_last_third_embeddings.csv"}, 
    ],
    target_systems=[
        'metabolites_unannotated',
    ],
    baseline=None,
    confounders=[],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/gait_only_no_confounders/',
    num_seeds=15,
    resume_seeds=False,  # Force full redo - ran with wrong file
    with_queue=True,
    ordinal_as_regression=True,
)

MORE_SEEDS_CONFIG = BuildResultsConfig(
    run_list = [
    # {JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant2_joint_stats/all_clean.csv
    {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
    "gait",
    {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"}
        #{"cycle_last_third": f"{JAFAR_ROOT}/results/best_embeddings/qjo95s37/all_last_third_embeddings.csv"}, 
    ],
    target_systems=[
        {"sleep_group": [
            'total_sleep_time', 'sleep_efficiency', 'sleep_latency', 'rem_latency',
            'total_rem_sleep_time', 'total_deep_sleep_time', 'total_light_sleep_time',
            'percent_of_rem_sleep_time', 'percent_of_deep_sleep_time', 'percent_of_light_sleep_time',
            'ahi', 'odi', 'rdi', 'csr_percent',
            'saturation_mean', 'saturation_min_value', 'hypoxic_burden',
            'heart_rate_mean_during_sleep', 'heart_rate_max_during_sleep', 'heart_rate_min_during_sleep'
        ]},
        'liver', 
        'nightingale',
        {'lifestyle_group': [
            # Sleep & sedentary
            'sleep_hours_in_24H', 'hours_using_computer_not_work', 'hours_watching_tv', 'hours_driving',
            # Work
            'work_hours_day', 'work_hours', 'work_days_a_week',
            # Outdoors
            'hours_outdoors_summer', 'hours_outdoors_winter',
            # Activity - continuous
            'walking_minutes_day', 'moderate_activity_minutes', 'vigorous_activity_minutes',
            'walking_10min_days_a_week', 'physical_activity_maderate_days_a_week',
            # Activity - categorical: manual work
            'manual_physical_work_Always', 'manual_physical_work_no', 'manual_physical_work_sometimes', 'manual_physical_work_usually',
            # Activity - categorical: stair climbing
            'climb_staires_tymes_a_day_1-5', 'climb_staires_tymes_a_day_6-10', 'climb_staires_tymes_a_day_11-15',
            'climb_staires_tymes_a_day_16-20', 'climb_staires_tymes_a_day_More than 20 times', 'climb_staires_tymes_a_day_Zero',
            # Activity - categorical: high exercise frequency
            'high_exercise_times_a_month_Every day', 'high_exercise_times_a_month_Four to five times a week',
            'high_exercise_times_a_month_Once a week', 'high_exercise_times_a_month_Two or three times a week',
            'high_exercise_times_a_month_Once in the last month', 'high_exercise_times_a_month_Two or three times in the last month',
            # Activity - categorical: high exercise duration
            'high_exercise_duration_Less than a quarter of an hour', 'high_exercise_duration_Between a quarter of an hour and half an hour',
            'high_exercise_duration_Between half an hour and an hour', 'high_exercise_duration_Between an hour and an hour and a half',
            'high_exercise_duration_Between an hour and a half and two hours', 'high_exercise_duration_Between 2 and 3 hours',
            'high_exercise_duration_Over 3 hours',
            # Activity - categorical: walking pace
            'usual_walking_pace_Slow', 'usual_walking_pace_Moderate', 'usual_walking_pace_Average', 'usual_walking_pace_Fast',
            # Alcohol
            'beer_cider_pints_week', 'white_wine_glasses_week', 'fortified_wine__glasses_week', 'liqueurs_measures_week', 'other_alcoholic_glasses_week'
        ]},
        'cardiovascular_system',
        'proteomics',
        'bone_density',
        'body_composition',
        'immune_system',
        'renal_function',
        'glycemic_status',
        'metabolomics',
    ],
    only_labels=[
        # sleep_group (1)
        'sleep_efficiency',
        # liver (2)
        'bt__alkaline_phosphatase', 'liver_viscosity',
        # nightingale (8)
        'XL_HDL_CE', 'L_VLDL_C', 'L_VLDL_FC', 'XL_HDL_C', 'L_HDL_FC', 
        'L_HDL_L', 'IDL_FC', 'XL_HDL_PL',
        # lifestyle_group (7)
        'hours_outdoors_summer', 'climb_staires_tymes_a_day_16-20',
        'high_exercise_duration_Between an hour and a half and two hours',
        'high_exercise_times_a_month_Every day', 'work_days_a_week',
        'manual_physical_work_sometimes', 'hours_driving',
        # cardiovascular_system (5)
        'p_mv_III', 'automorph_artery_fractal_dimension', 't_mv_V3', 
        't_mv_III', 'l_abi',
        # proteomics (9)
        'PTPRN2', 'BAX', 'ITGA11', 'RGS8', 'GRPEL1', 'BCAN', 'ACY3', 
        'ACVRL1', 'HSPB6',
        # bone_density (8)
        'femur_total_diff_area', 'femur_right_upper_neck_area', 
        'femur_upper_neck_mean_bmc', 'femur_left_wards_bmc', 
        'femur_right_neck_bmc', 'femur_troch_diff_area', 
        'femur_right_wards_bmc', 'femur_left_neck_bmd',
        # body_composition (1)
        'total_scan_vat_volume',
        # immune_system (2)
        'bt__lymphocytes_%', 'bt__wbc',
        # renal_function (1)
        'bt__sodium',
        # glycemic_status (1)
        'iglu_m_value',
    ],
    baseline={'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},
    confounders=['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/long_vs_short_vs_features_vs_baseline_confounders2/',
    num_seeds=50,
    resume_seeds=False,  # Force full redo - ran with wrong file
    with_queue=True,
)

# Movement data features - split by activity type
MOVEMENT_DATA_CONFIG = BuildResultsConfig(
    run_list=[
        {'tm_3kmh': f"{JAFAR_ROOT}/skeleton_data/features/movement_data_TM3.csv"},
        {'self_selected_gait_speed': f"{JAFAR_ROOT}/skeleton_data/features/movement_data_TMS.csv"},
        {'rombergs_closed': f"{JAFAR_ROOT}/skeleton_data/features/movement_data_rombergs_closed.csv"},
        {'sit_to_stand': f"{JAFAR_ROOT}/skeleton_data/features/movement_data_sit_to_stand.csv"},
        {'stationary': f"{JAFAR_ROOT}/skeleton_data/features/movement_data_stationary.csv"},
    ],
    target_systems=[
        {'age_bmi_vat': ['age', 'bmi', 'total_scan_vat_area']},
    ],
    baseline=None,  # No baseline
    confounders=['gender', 'height'],
    save_dir=f'{JAFAR_ROOT}/results/movement_data_features/',
    with_queue=True,
)


# Lifestyle ordinal - clean stage matching (no fallback to baseline)
# Tests gait prediction of self-reported lifestyle with proper ordinal encoding
# Uses merge_closest_research_stage=False to avoid ~20% noise from mismatched stages
LIFESTYLE_ORDINAL_CONFIG = BuildResultsConfig(
    run_list=[
        {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
        "gait",
        {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"},
    ],
    target_systems=[
        {'lifestyle_ordinal': f"{GENIE_ROOT}/Analyses/10K_Trajectories/body_systems/lifestyle_ordinal.csv"},
        {"exercise_label": f"{JAFAR_ROOT}/skeleton_data/subject_data/wearable_exercise_labels.csv"},
        "mental"
    ],
    baseline={'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},
    confounders=['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    target_column_descriptions={},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/long_vs_short_vs_features_vs_baseline_confounders2/',
    num_seeds=15,
    merge_closest_research_stage=False,  # IMPORTANT: exact stage matching only
    ordinal_as_regression=True,  # Treat ordinal targets as regression (faster)
    with_queue=True,
)

HIGH_LEVEL_DIET_CONFIG = BuildResultsConfig(
    run_list=[
        {"long_seq": f"{JAFAR_ROOT}/results/best_embeddings/lqc2exvo_long_all/variant5_merged_groups_percentile/all_cleaned-visits_median_pooled.csv"},
        "gait",
        {'single_cycle': f"{JAFAR_ROOT}/results/best_embeddings/8ze9xt8e_cls_token/variant1_all_pooled_cleaned/all.csv"},
    ],
    target_systems=[
        "high_level_diet_with_stage",
    ],
    baseline={'Age_Gender_BMI_height_VAT': ['age', 'gender', 'bmi', "height", "total_scan_vat_area"]},
    confounders=['age', 'gender', 'bmi', "height", "total_scan_vat_area"],
    run_column_descriptions={"activity": "classification", "seq_idx": "ordinalf"},
    target_column_descriptions={},
    save_dir=f'{JAFAR_ROOT}/results/embeddings_results/long_vs_short_vs_features_vs_baseline_confounders2/',
    num_seeds=15,
    merge_closest_research_stage=False,  # IMPORTANT: exact stage matching only
    ordinal_as_regression=True,  # Treat ordinal targets as regression (faster)
    with_queue=True,
)

# Glycoproteomics (Shira) - GP1-GP39, no gait embeddings
# Sep_groups, age, bmi as confounders and baseline; baseline stage only.
# glyco_baseline_confounders.csv contains Sep_groups + age + bmi merged at subject level,
# so all 3 are in one registered temp system (avoids Sep_groups not being in body_systems).
_GLYCO_CSV = f"{SEGAL_GENIE_ROOT}/Analyses/shiragel/data/glyco_ready_run.csv"
_GLYCO_CONFOUNDERS_CSV = f"{SEGAL_GENIE_ROOT}/Analyses/shiragel/data/glyco_baseline_confounders.csv"
GLYCO_CONFIG = BuildResultsConfig(
    run_list=[
        {'glyco': _GLYCO_CSV},
    ],
    target_systems=[
        'blood_tests_lipids', 'body_composition', 'bone_density', 'cardiovascular_system',
        'frailty', 'glycemic_status', 'hematopoietic', 'immune_system', 'liver',
        'mental', 'metabolites_annotated', 'metabolites_unannotated', 'microbiome',
        'nightingale', 'proteomics', 'renal_function', 'sleep', 'high_level_diet_with_stage', "gait",
    ],
    baseline={'glyco_confounders': _GLYCO_CONFOUNDERS_CSV},  # Sep_groups + age + bmi
    confounders=['Sep_groups', 'age', 'bmi'],  # Sep_groups from glyco_confounders, age/bmi from body systems
    save_dir=f'{SEGAL_GENIE_ROOT}/Analyses/shiragel/results/',
    merge_closest_research_stage=False,                      # baseline stage only
    with_queue=True,
)
