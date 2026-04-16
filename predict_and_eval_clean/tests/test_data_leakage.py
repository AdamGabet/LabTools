"""
Data leakage tests for predict_and_eval_clean.

All tests use synthetic DataFrames — no HPP data required.
The core invariant: splits must be on RegistrationCode (subject) level.
A subject's rows must NEVER appear in both train and test within the same fold.
"""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from predict_and_eval_clean.ids_folds import ids_folds, id_fold_with_stratified_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multiindex_df(n_subjects: int, stages_per_subject: int = 2,
                        seed: int = 0) -> pd.DataFrame:
    """
    Create a synthetic DataFrame with a (RegistrationCode, research_stage) MultiIndex.
    Each subject has `stages_per_subject` rows.
    """
    rng = np.random.default_rng(seed)
    subjects = [f"10K_{i:010d}" for i in range(n_subjects)]
    stages = list(range(1, stages_per_subject + 1))
    idx = pd.MultiIndex.from_product([subjects, stages],
                                     names=["RegistrationCode", "research_stage"])
    df = pd.DataFrame(
        {"value": rng.standard_normal(len(idx)),
         "label": rng.integers(0, 2, size=len(idx))},
        index=idx,
    )
    return df


def _extract_train_test_subjects(fold):
    """Return (set of train subjects, set of test subjects) from a single fold."""
    train_ids, test_ids = fold
    return set(train_ids), set(test_ids)


# ---------------------------------------------------------------------------
# ids_folds tests
# ---------------------------------------------------------------------------

class TestNoSubjectLeakage:
    """Core invariant: no RegistrationCode appears in both train AND test."""

    def test_no_overlap_single_seed(self):
        df = _make_multiindex_df(100, stages_per_subject=2)
        folds = ids_folds(df, seeds=[0], n_splits=5)

        for seed_folds in folds:
            for fold in seed_folds:
                train_subjects, test_subjects = _extract_train_test_subjects(fold)
                overlap = train_subjects & test_subjects
                assert len(overlap) == 0, (
                    f"Subject overlap between train and test: {overlap}"
                )

    def test_no_overlap_multiple_seeds(self):
        df = _make_multiindex_df(80, stages_per_subject=3)
        folds = ids_folds(df, seeds=range(5), n_splits=4)

        for seed_idx, seed_folds in enumerate(folds):
            for fold_idx, fold in enumerate(seed_folds):
                train_subjects, test_subjects = _extract_train_test_subjects(fold)
                overlap = train_subjects & test_subjects
                assert len(overlap) == 0, (
                    f"Seed {seed_idx}, fold {fold_idx}: subject overlap {overlap}"
                )

    def test_all_subjects_covered_per_seed(self):
        """Every subject must appear in exactly one test fold per seed."""
        n_subjects = 60
        df = _make_multiindex_df(n_subjects, stages_per_subject=2)
        folds = ids_folds(df, seeds=[42], n_splits=5)

        for seed_folds in folds:
            test_subjects_seen = set()
            for fold in seed_folds:
                _, test_subjects = _extract_train_test_subjects(fold)
                # No subject should appear in multiple test folds
                overlap = test_subjects_seen & test_subjects
                assert len(overlap) == 0, f"Subject in multiple test folds: {overlap}"
                test_subjects_seen |= test_subjects

            all_subjects = set(df.index.get_level_values("RegistrationCode").unique())
            assert test_subjects_seen == all_subjects, (
                "Not all subjects appeared in exactly one test fold"
            )


class TestSubjectRowsStayTogether:
    """All rows for a subject must land in the same fold (train or test)."""

    def test_multi_stage_rows_not_split(self):
        """Subject with multiple research stages must not be split across folds."""
        df = _make_multiindex_df(50, stages_per_subject=4)
        folds = ids_folds(df, seeds=[0], n_splits=5)

        for seed_folds in folds:
            for fold in seed_folds:
                train_subjects, test_subjects = _extract_train_test_subjects(fold)
                # Verify directly: for each subject, all their rows land in one set
                for subj in test_subjects:
                    assert subj not in train_subjects, (
                        f"Subject {subj} split across train and test"
                    )

    def test_single_stage_subjects(self):
        """Works correctly when every subject has exactly one row."""
        df = _make_multiindex_df(100, stages_per_subject=1)
        folds = ids_folds(df, seeds=[7], n_splits=5)

        for seed_folds in folds:
            for fold in seed_folds:
                train_subjects, test_subjects = _extract_train_test_subjects(fold)
                assert len(train_subjects & test_subjects) == 0


class TestFoldSizes:
    """Folds should be approximately equal in size."""

    def test_n_splits_is_respected(self):
        df = _make_multiindex_df(100, stages_per_subject=2)
        for n_splits in [3, 5, 10]:
            folds = ids_folds(df, seeds=[0], n_splits=n_splits)
            assert len(folds[0]) == n_splits, (
                f"Expected {n_splits} folds, got {len(folds[0])}"
            )

    def test_approximately_equal_fold_sizes(self):
        """Test fold sizes are within 1 subject of each other."""
        n_subjects = 100
        df = _make_multiindex_df(n_subjects, stages_per_subject=2)
        folds = ids_folds(df, seeds=[0], n_splits=5)

        test_sizes = [len(fold[1]) for fold in folds[0]]
        assert max(test_sizes) - min(test_sizes) <= 1, (
            f"Unbalanced folds: {test_sizes}"
        )

    def test_different_seeds_give_different_splits(self):
        """Different seeds must produce different fold assignments."""
        df = _make_multiindex_df(60, stages_per_subject=2)
        folds_a = ids_folds(df, seeds=[0], n_splits=5)
        folds_b = ids_folds(df, seeds=[1], n_splits=5)

        # Compare test sets of first fold from each seed
        test_a = set(folds_a[0][0][1])
        test_b = set(folds_b[0][0][1])
        assert test_a != test_b, "Different seeds produced identical splits"


# ---------------------------------------------------------------------------
# id_fold_with_stratified_threshold
# ---------------------------------------------------------------------------

class TestStratifiedFolds:
    """Stratified folds must also respect subject-level splitting."""

    def test_stratified_no_subject_overlap(self):
        df = _make_multiindex_df(100, stages_per_subject=2)
        # id_fold_with_stratified_threshold takes a df whose first column is the label
        subjects = df.index.get_level_values("RegistrationCode").unique()
        rng = np.random.default_rng(0)
        label_values = rng.integers(0, 2, len(subjects))
        label_df = pd.DataFrame(
            {"disease": label_values},
            index=pd.MultiIndex.from_product(
                [subjects, [1]], names=["RegistrationCode", "research_stage"]
            )
        )

        folds = id_fold_with_stratified_threshold(
            label_df, seeds=[0], n_splits=5,
            stratified_threshold=0.5
        )

        for seed_folds in folds:
            for fold in seed_folds:
                train_ids, test_ids = fold
                overlap = set(train_ids) & set(test_ids)
                assert len(overlap) == 0, f"Stratified fold has subject overlap: {overlap}"


# ---------------------------------------------------------------------------
# Row-level split detection
# ---------------------------------------------------------------------------

class TestRowLevelSplitIsForbidden:
    """
    Demonstrates that row-level splits (like sklearn train_test_split on rows)
    would cause data leakage. These tests document the expected CORRECT behavior.
    """

    def test_subject_level_split_is_deterministic(self):
        """Same seed always produces the same subject-level split."""
        df = _make_multiindex_df(80, stages_per_subject=3)
        folds_1 = ids_folds(df, seeds=[99], n_splits=5)
        folds_2 = ids_folds(df, seeds=[99], n_splits=5)

        for fold_1, fold_2 in zip(folds_1[0], folds_2[0]):
            assert set(fold_1[0]) == set(fold_2[0]), "Train sets differ for same seed"
            assert set(fold_1[1]) == set(fold_2[1]), "Test sets differ for same seed"

    def test_train_covers_majority_of_subjects(self):
        """For 5-fold CV, train set should contain ~80% of subjects."""
        n_subjects = 100
        df = _make_multiindex_df(n_subjects, stages_per_subject=2)
        folds = ids_folds(df, seeds=[0], n_splits=5)

        for fold in folds[0]:
            train_subjects, test_subjects = _extract_train_test_subjects(fold)
            train_ratio = len(train_subjects) / n_subjects
            assert 0.75 <= train_ratio <= 0.85, (
                f"Train ratio {train_ratio:.2f} outside expected range for 5-fold CV"
            )
