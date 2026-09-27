"""
Real pytest tests for the multi-objective molecular property loss pipeline.
Runs a fast, small version of the actual pipeline (real data generation,
real model training, real loss computations) -- nothing here is mocked out.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_compound_data import generate, train_val_test_split, PROPERTY_NAMES, N_COMPOUNDS
from loss_functions import (
    LOSS_REGISTRY,
    PROPERTY_LOSS_ASSIGNMENT,
    CORRELATION_PAIRS,
    PARETO_PAIRS,
    ALL_LOSS_NAMES,
    UncertaintyWeightedLoss,
    ParetoTradeoffLoss,
)
from train_multi_objective import search


@pytest.fixture(scope="module")
def data():
    df = generate()
    return df, train_val_test_split(df)


def test_compound_count(data):
    df, _ = data
    assert len(df) == 560 == N_COMPOUNDS


def test_property_count(data):
    df, _ = data
    assert len(PROPERTY_NAMES) == 12
    for p in PROPERTY_NAMES:
        assert p in df.columns


def test_eight_loss_functions_registered():
    assert len(LOSS_REGISTRY) == 8
    assert set(ALL_LOSS_NAMES) == set(LOSS_REGISTRY.keys())


def test_all_12_properties_covered_by_assignment():
    assert set(PROPERTY_LOSS_ASSIGNMENT.keys()) == set(PROPERTY_NAMES)
    for p, losses in PROPERTY_LOSS_ASSIGNMENT.items():
        assert len(losses) >= 1
        for ln in losses:
            assert ln in LOSS_REGISTRY


def test_all_8_losses_actually_used_somewhere():
    used = set()
    for losses in PROPERTY_LOSS_ASSIGNMENT.values():
        used.update(losses)
    assert used == set(ALL_LOSS_NAMES)


def test_each_loss_function_computes_real_score_on_real_data(data):
    """Exercise each of the 8 losses at least once with real data and check
    a real, finite scalar score/gradient-signal comes out."""
    df, (train_df, val_df, test_df) = data
    rng = np.random.default_rng(0)
    n = len(train_df)

    y_reg_true = train_df["solubility"].values
    y_reg_pred = y_reg_true + rng.normal(0, 0.3, n)
    huber = LOSS_REGISTRY["huber_loss"]
    v = huber.evaluate(y_reg_true, y_reg_pred)
    w = huber.sample_weights(y_reg_true, y_reg_pred)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,) and np.all(np.isfinite(w))

    y_clf_true = train_df["toxicity_class"].values
    y_prob = np.clip(rng.uniform(0, 1, n), 1e-3, 1 - 1e-3)
    focal = LOSS_REGISTRY["focal_loss"]
    v = focal.evaluate(y_clf_true, y_prob)
    w = focal.sample_weights(y_clf_true, y_prob)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,)

    y_rank_true = train_df["melting_point"].values
    y_rank_pred = y_rank_true + rng.normal(0, 0.5, n)
    rank = LOSS_REGISTRY["rank_consistency_loss"]
    v = rank.evaluate(y_rank_true, y_rank_pred)
    w = rank.sample_weights(y_rank_true, y_rank_pred)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,)

    a_true, b_true = train_df["melting_point"].values, train_df["synthetic_accessibility"].values
    a_pred = a_true + rng.normal(0, 0.4, n)
    b_pred = b_true + rng.normal(0, 0.4, n)
    corr = LOSS_REGISTRY["correlation_consistency_loss"]
    v = corr.evaluate(a_true, a_pred, b_true, b_pred)
    w = corr.sample_weights(a_true, a_pred, b_true, b_pred)
    assert np.isfinite(v)
    assert w.shape == (n,)

    ba_true = train_df["binding_affinity"].values
    ba_pred = ba_true + rng.normal(0, 0.5, n)
    varpen = LOSS_REGISTRY["variance_penalty_loss"]
    v = varpen.evaluate(ba_true, ba_pred)
    w = varpen.sample_weights(ba_true, ba_pred)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,)

    perm_true = train_df["permeability"].values
    perm_pred = perm_true + rng.normal(0, 0.4, n)
    tol = LOSS_REGISTRY["tolerance_band_loss"]
    v = tol.evaluate(perm_true, perm_pred)
    w = tol.sample_weights(perm_true, perm_pred)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,)

    unc = UncertaintyWeightedLoss.uncertainty_from_residual_bins(ba_pred, ba_true)
    uw = LOSS_REGISTRY["uncertainty_weighted_loss"]
    v = uw.evaluate(ba_true, ba_pred, unc)
    w = uw.sample_weights(unc)
    assert np.isfinite(v)
    assert w.shape == (n,)

    pareto = ParetoTradeoffLoss()
    sol_true, logp_true = train_df["solubility"].values, train_df["logp"].values
    pareto.fit_frontier(("solubility", "logp"), sol_true, logp_true)
    sol_pred = sol_true + rng.normal(0, 0.4, n)
    logp_pred = logp_true + rng.normal(0, 0.4, n)
    v = pareto.evaluate(("solubility", "logp"), sol_pred, logp_pred)
    w = pareto.sample_weights(("solubility", "logp"), sol_true, sol_pred, logp_true, logp_pred)
    assert np.isfinite(v) and v >= 0
    assert w.shape == (n,)


@pytest.fixture(scope="module")
def small_search_result(data):
    _, splits = data
    naive_best, optimized_best, trajectory = search(splits, seed=123, n_naive=2, n_extra=3)
    return naive_best, optimized_best, trajectory


def test_optimized_beats_or_matches_naive(small_search_result):
    naive_best, optimized_best, _ = small_search_result
    assert optimized_best["agg_val_accuracy"] >= naive_best["agg_val_accuracy"] - 1e-9


def test_improvement_in_reasonable_band():
    """Full-size search (matching run_all.py's settings) should land the
    real % improvement of optimized-vs-naive in a plausible band. This uses
    the full search budget (slower) to reflect the actual reported result,
    not the tiny smoke-test search above."""
    from generate_compound_data import generate, train_val_test_split, SEED

    df = generate()
    splits = train_val_test_split(df)
    naive_best, optimized_best, _ = search(splits, seed=SEED, n_naive=6, n_extra=30)
    improvement_pct = (
        (optimized_best["agg_test_accuracy"] - naive_best["agg_test_accuracy"])
        / naive_best["agg_test_accuracy"] * 100.0
    )
    assert optimized_best["agg_test_accuracy"] >= naive_best["agg_test_accuracy"]
    assert 15.0 <= improvement_pct <= 55.0, f"improvement {improvement_pct:.1f}% out of band"
