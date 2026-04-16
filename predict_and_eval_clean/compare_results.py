import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from .load_feature_df import get_column_info
from .fix_pvals import fix_pvals

MIN_SEEDS_FOR_WILCOX = 6


def compare_and_collect_results(main_dir, baseline_features_system_name=None, 
                                main_reg_score='r2', main_reg_pvalue='pearson_pvalue', 
                                main_clas_score='auc', 
                                main_ordinal_score='spearman_r', main_ordinal_pvalue='spearman_pvalue',
                                significance_threshold=0.05, skip_systems=None,
                                wilcox_exclude_by='pvalue', wilcox_exclude_threshold=0.05):
    """
    Compare feature systems to baseline and collect results.
    If baseline_features_system_name is None, outputs absolute scores without delta/wilcox.
    For <6 seeds: only compares means, wilcox/pvalue/is_significant are NaN.
    Also applies FDR correction to p-values in metrics.csv files.
    
    Args:
        skip_systems: List of target system names to skip.
        wilcox_exclude_by: 'pvalue' to exclude by FDR-corrected p-value, 'score' to exclude by score threshold.
        wilcox_exclude_threshold: Threshold for exclusion (0.05 for pvalue, 0.02 for r2).
    """
    skip_systems = set(skip_systems or [])
    
    # Collect all metrics directories and apply FDR correction (Benjamini-Hochberg)
    all_dirs = _collect_metrics_dirs(main_dir, skip_systems)
    fix_pvals(all_dirs)
    
    all_results = []
    label_info = []
    
    for target_system in os.listdir(main_dir):
        if target_system in skip_systems:
            continue
        target_system_path = os.path.join(main_dir, target_system)
        if not os.path.isdir(target_system_path):
            continue
        
        for label_name in os.listdir(target_system_path):
            label_path = os.path.join(target_system_path, label_name)
            if not os.path.isdir(label_path):
                continue
            
            label_results = _process_label_directory(
                label_path, target_system, label_name, baseline_features_system_name,
                main_reg_score, main_reg_pvalue, main_clas_score,
                main_ordinal_score, main_ordinal_pvalue
            )
            
            if label_results:
                all_results.extend(label_results)
                label_info.extend([label_path] * len(label_results))
    
    if not all_results:
        return None
    
    df = pd.DataFrame(all_results)
    df['_label_path'] = label_info
    
    # Apply wilcox FDR corrections only if baseline comparison was done
    if baseline_features_system_name and 'wilcox_pvalue' in df.columns:
        # Determine which rows should be excluded from Wilcoxon (mark as insignificant directly)
        if wilcox_exclude_by == 'pvalue':
            # Use FDR-corrected score p-value
            pval_col = 'score_pvalue_fdr' if 'score_pvalue_fdr' in df.columns else 'score_pvalue'
            exclude_mask = df[pval_col] >= wilcox_exclude_threshold
        else:  # score
            exclude_mask = df['score'] < wilcox_exclude_threshold
        
        # Set wilcox_pvalue to NaN for excluded rows (won't be included in FDR correction)
        df.loc[exclude_mask, 'wilcox_pvalue'] = np.nan
        
        df = _apply_fdr_correction_to_wilcox(df)

        # is_significant: False for excluded rows, NaN if wilcox wasn't run for other reasons
        df['is_significant'] = np.where(exclude_mask, False,
                                         np.where(df['wilcox_pvalue'].isna(), np.nan, 
                                                  df['wilcox_pvalue_fdr'] < significance_threshold))
    
    # Reorder columns: first columns depend on whether baseline exists
    if baseline_features_system_name:
        first_cols = ['system', 'label', 'score_type', 'score', 'delta', 'is_better', 'is_significant']
        sort_col = 'delta'
    else:
        first_cols = ['system', 'label', 'score_type', 'score', 'model']
        sort_col = 'score'
    other_cols = [col for col in df.columns if col not in first_cols and col != '_label_path']
    df = df[[c for c in first_cols if c in df.columns] + other_cols + ['_label_path']]
    
    df_sorted = df.sort_values(['system', sort_col], ascending=[True, False])
    
    # Save main summary
    main_summary = df_sorted.drop(columns=['_label_path'])
    main_summary.to_csv(os.path.join(main_dir, 'all_comparisons.csv'), index=False)
    
    # Save per-target-system summaries
    for target_system in df['system'].unique():
        system_df = df[df['system'] == target_system].drop(columns=['_label_path'])
        system_df.sort_values(sort_col, ascending=False).to_csv(
            os.path.join(main_dir, target_system, 'system_summary.csv'), index=False)
    
    # Save per-label summaries
    for label_path in df['_label_path'].unique():
        label_df = df[df['_label_path'] == label_path].drop(columns=['_label_path'])
        label_df.sort_values(sort_col, ascending=False).to_csv(
            os.path.join(label_path, 'comparison_summary.csv'), index=False)
    
    return main_summary


def _process_label_directory(label_path, target_system, label_name, baseline_name,
                             main_reg_score, main_reg_pvalue, main_clas_score,
                             main_ordinal_score, main_ordinal_pvalue):
    """Process a single label directory."""
    feature_systems = {}
    baseline_data = None
    
    for feature_name in os.listdir(label_path):
        feature_path = os.path.join(label_path, feature_name)
        metrics_file = os.path.join(feature_path, 'metrics.csv')
        if not os.path.isdir(feature_path) or not os.path.exists(metrics_file):
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        if baseline_name and feature_name == baseline_name:
            baseline_data = metrics_df
        else:
            feature_systems[feature_name] = metrics_df
    
    # If baseline specified but not found, skip this label
    if baseline_name and baseline_data is None:
        return []
    
    if not feature_systems:
        return []
    
    # Use first feature system to determine score type if no baseline
    ref_df = baseline_data if baseline_data is not None else next(iter(feature_systems.values()))
    
    # Determine score type
    is_clas = main_clas_score in ref_df.columns
    is_regression = (main_reg_score in ref_df.columns or 'pearson_r' in ref_df.columns) and not is_clas
    is_ordinal = main_ordinal_score in ref_df.columns and not is_clas and not is_regression
    
    if is_clas:
        main_score, main_pvalue, score_type = main_clas_score, None, 'auc'
    elif is_regression:
        main_score, main_pvalue, score_type = main_reg_score, main_reg_pvalue, main_reg_score
    elif is_ordinal:
        main_score, main_pvalue, score_type = main_ordinal_score, main_ordinal_pvalue, 'spearman'
    else:
        main_score, main_pvalue, score_type = main_reg_score, main_reg_pvalue, main_reg_score
    
    col_info = get_column_info(label_name)
    
    results = []
    for feature_name, feature_df in feature_systems.items():
        genders = ['all']
        if f'male_{main_score}' in feature_df.columns:
            genders.extend(['male', 'female'])
        
        for gender in genders:
            row = _build_result_row(
                feature_name, feature_df, baseline_name, baseline_data,
                target_system, label_name, gender, main_score, main_pvalue,
                score_type, col_info
            )
            if row:
                results.append(row)
    
    return results


def _build_result_row(feature_name, feature_df, baseline_name, baseline_df,
                      target_system, label_name, gender, main_score, main_pvalue,
                      score_type, col_info):
    """Build result row, with or without baseline comparison."""
    score_col = main_score if gender == 'all' else f'{gender}_{main_score}'
    
    if score_col not in feature_df.columns:
        return None
    
    # Check baseline has score col if baseline exists
    if baseline_df is not None and score_col not in baseline_df.columns:
        return None
    
    feat_scores = feature_df[score_col].values
    n_seeds = len(feat_scores)
    feat_mean = np.mean(feat_scores)
    
    # Score p-value (None for classification)
    score_pvalue = np.nan
    if main_pvalue:
        pval_col = main_pvalue if gender == 'all' else f'{gender}_{main_pvalue}'
        if pval_col in feature_df.columns:
            score_pvalue = np.mean(feature_df[pval_col].values)
    
    n_subjects = int(feature_df['n_subjects'].mean()) if 'n_subjects' in feature_df.columns else np.nan
    n_positive = int(feature_df['n_positive'].mean()) if 'n_positive' in feature_df.columns else np.nan
    
    result = {
        'system': target_system,
        'label': label_name,
        'score_type': score_type,
        'score': feat_mean,
        'model': feature_name,
        'gender': gender,
        'label_type': col_info['type'],
        'description': col_info['description'],
        'score_pvalue': score_pvalue,
        'n_seeds': n_seeds,
        'n_subjects': n_subjects,
        'n_positive': n_positive
    }
    
    # Add baseline comparison fields only if baseline exists
    if baseline_df is not None:
        base_scores = baseline_df[score_col].values
        base_mean = np.mean(base_scores)
        delta = feat_mean - base_mean
        
        # Wilcoxon only with enough seeds
        wilcox_pval = np.nan
        if n_seeds >= MIN_SEEDS_FOR_WILCOX:
            try:
                _, wilcox_pval = wilcoxon(feat_scores, base_scores, alternative='greater')
            except Exception:
                pass
        
        result.update({
            'delta': delta,
            'is_better': delta > 0,
            'baseline_model': baseline_name,
            'baseline_score': base_mean,
            'wilcox_pvalue': wilcox_pval,
        })
    
    return result


def _apply_fdr_correction_to_wilcox(df):
    """Apply FDR correction to Wilcoxon p-values per (gender, model) group.
    
    FDR is applied separately for each combination because:
    - Each gender (all, male, female) is a separate analysis
    - Each model/features is a separate hypothesis family
    """
    df['wilcox_pvalue_fdr'] = np.nan
    
    for (gender, model), group in df.groupby(['gender', 'model']):
        mask = (df['gender'] == gender) & (df['model'] == model)
        pvals = df.loc[mask, 'wilcox_pvalue'].values
        valid = ~np.isnan(pvals)
        
        if valid.any():
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid], _, _ = multipletests(pvals[valid], method='fdr_bh')
            df.loc[mask, 'wilcox_pvalue_fdr'] = corrected
    
    return df


def _collect_metrics_dirs(main_dir, skip_systems=None):
    """Collect all directories containing metrics.csv files."""
    skip_systems = skip_systems or set()
    all_dirs = []
    for target_system in os.listdir(main_dir):
        if target_system in skip_systems:
            continue
        target_path = os.path.join(main_dir, target_system)
        if not os.path.isdir(target_path):
            continue
        for label_name in os.listdir(target_path):
            label_path = os.path.join(target_path, label_name)
            if not os.path.isdir(label_path):
                continue
            for feature_name in os.listdir(label_path):
                feature_path = os.path.join(label_path, feature_name)
                if os.path.isdir(feature_path) and os.path.exists(os.path.join(feature_path, 'metrics.csv')):
                    all_dirs.append(feature_path)
    return all_dirs
