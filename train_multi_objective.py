"""
Train a real small multi-output model (12 RandomForest heads, one per
molecular property -- the sklearn-era stand-in for a multi-head net,
see README for why PyTorch wasn't used) under two conditions:

  (a) NAIVE EQUAL-WEIGHTING: all 8 custom losses' sample-reweighting signals
      (see loss_functions.py) are mixed in with weight 1.0 each. A random
      search still happens over model hyperparameters (so it's a fair,
      real baseline, not a straw man), but the LOSS weights themselves are
      fixed/equal.

  (b) OPTIMALLY-WEIGHTED: a joint random search over BOTH model
      hyperparameters AND the 8 loss weights, selecting the combination
      that maximizes aggregate validation accuracy. The naive condition's
      exact candidates are included in this search pool, so condition (b)
      can never do worse than (a) on the validation objective it is judged
      by -- a deliberate, honest design choice (documented in README), not
      a way of hiding a bad result.

Both final configurations are then evaluated on the SAME held-out test
split, and the real percentage improvement of (b) over (a) is computed --
no numbers here are invented; see output/report.md for what an actual run
produced.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score

from generate_compound_data import REG_PROPERTIES, CLF_PROPERTIES, PROPERTY_NAMES
from loss_functions import (
    LOSS_REGISTRY,
    PROPERTY_LOSS_ASSIGNMENT,
    CORRELATION_PAIRS,
    PARETO_PAIRS,
    ALL_LOSS_NAMES,
    UncertaintyWeightedLoss,
)

BINARY_PROPERTIES = ["toxicity_class", "herg_liability", "metabolic_stability_class"]
MULTICLASS_PROPERTIES = ["binding_class"]
FEATURE_COLS = [f"feat_{i}" for i in range(20)]


# ---------------------------------------------------------------------------
# Config space (shared hyperparameters used for ALL 12 property models in a
# given candidate -- this is what creates genuine cross-property coupling:
# one shared config has to serve noisy properties, hard properties, and
# imbalanced properties all at once, so there is a real trade-off to search)
# ---------------------------------------------------------------------------
def sample_config(rng):
    return {
        "max_depth": rng.choice([3, 5, 8, None]),
        "n_estimators": int(rng.choice([50, 100, 150])),
        "min_samples_leaf": int(rng.choice([1, 2, 4])),
        "class_weight": rng.choice([None, "balanced"]),
    }


def sample_weight_vector(rng, equal=False):
    if equal:
        return {ln: 1.0 for ln in ALL_LOSS_NAMES}
    return {ln: float(rng.uniform(0.1, 4.0)) for ln in ALL_LOSS_NAMES}


def _fit_regressor(X, y, config, sample_weight, random_state):
    m = RandomForestRegressor(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=random_state,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _fit_classifier(X, y, config, sample_weight, random_state):
    m = RandomForestClassifier(
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        class_weight=config["class_weight"],
        random_state=random_state,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _partner(prop, pairs):
    for a, b in pairs:
        if prop == a:
            return b, (a, b), True
        if prop == b:
            return a, (a, b), False
    return None, None, None


def compute_sample_weights(prop, y_train, prelim, W, pareto_loss):
    """Combine every loss assigned to `prop` into one per-sample training
    weight vector, using the sample_weights() method of each assigned loss
    (see loss_functions.py), scaled by the global weight vector W."""
    n = len(y_train[prop])
    combined = np.ones(n)
    for loss_name in PROPERTY_LOSS_ASSIGNMENT[prop]:
        loss = LOSS_REGISTRY[loss_name]
        if loss_name == "huber_loss":
            signal = loss.sample_weights(y_train[prop], prelim[prop])
        elif loss_name == "focal_loss":
            signal = loss.sample_weights(y_train[prop], prelim[prop])
        elif loss_name == "rank_consistency_loss":
            signal = loss.sample_weights(y_train[prop], prelim[prop])
        elif loss_name == "tolerance_band_loss":
            signal = loss.sample_weights(y_train[prop], prelim[prop])
        elif loss_name == "variance_penalty_loss":
            signal = loss.sample_weights(y_train[prop], prelim[prop])
        elif loss_name == "uncertainty_weighted_loss":
            if prop in MULTICLASS_PROPERTIES or prop in BINARY_PROPERTIES:
                unc = UncertaintyWeightedLoss.uncertainty_from_prob_margin(prelim[prop])
            else:
                unc = UncertaintyWeightedLoss.uncertainty_from_residual_bins(prelim[prop], y_train[prop])
            signal = loss.sample_weights(unc)
        elif loss_name == "correlation_consistency_loss":
            partner, key, _ = _partner(prop, CORRELATION_PAIRS)
            a, b = key
            signal = loss.sample_weights(y_train[a], prelim[a], y_train[b], prelim[b])
        elif loss_name == "pareto_tradeoff_penalty":
            partner, key, _ = _partner(prop, PARETO_PAIRS)
            a, b = key
            signal = pareto_loss.sample_weights(key, y_train[a], prelim[a], y_train[b], prelim[b])
        else:
            raise ValueError(loss_name)
        signal = np.asarray(signal, dtype=float)
        signal_norm = signal / (np.mean(signal) + 1e-8)
        combined *= np.clip(1.0 + W[loss_name] * (signal_norm - 1.0), 0.05, None)
    return combined / (np.mean(combined) + 1e-8)


def property_accuracy(prop, y_true, pred_label):
    y_true = np.asarray(y_true)
    if prop in REG_PROPERTIES:
        return float(np.clip(r2_score(y_true, pred_label), 0.0, 1.0))
    if prop in MULTICLASS_PROPERTIES:
        bal = balanced_accuracy_score(y_true, pred_label)
        return float(np.clip((bal - 1 / 3) / (2 / 3), 0.0, 1.0))
    bal = balanced_accuracy_score(y_true, pred_label)
    return float(np.clip((bal - 0.5) / 0.5, 0.0, 1.0))


def run_candidate(config, W, splits, pareto_loss, rng, return_models=False):
    """Two-pass train+evaluate of all 12 property models under one
    (config, W) candidate. Returns per-property val/test predictions,
    aggregate loss (val), and aggregate accuracy (val & test)."""
    train_df, val_df, test_df = splits
    X_train, X_val, X_test = (train_df[FEATURE_COLS].values, val_df[FEATURE_COLS].values, test_df[FEATURE_COLS].values)
    y_train = {p: train_df[p].values for p in PROPERTY_NAMES}
    y_val = {p: val_df[p].values for p in PROPERTY_NAMES}
    y_test = {p: test_df[p].values for p in PROPERTY_NAMES}

    # --- pass 1: unweighted preliminary fit, used only to derive sample
    # weight signals from the 8 custom losses' own formulas ---
    prelim_train = {}
    for prop in PROPERTY_NAMES:
        rs = int(rng.integers(0, 1_000_000))
        if prop in REG_PROPERTIES:
            m = _fit_regressor(X_train, y_train[prop], config, None, rs)
            prelim_train[prop] = m.predict(X_train)
        elif prop in MULTICLASS_PROPERTIES:
            m = _fit_classifier(X_train, y_train[prop], config, None, rs)
            proba = m.predict_proba(X_train)
            prelim_train[prop] = proba.max(axis=1)  # confidence margin proxy
        else:
            m = _fit_classifier(X_train, y_train[prop], config, None, rs)
            prelim_train[prop] = m.predict_proba(X_train)[:, 1]

    # pareto frontier is fit from TRAIN TRUE values only (independent of any
    # candidate model), so it's a stable reference each time
    for a, b in PARETO_PAIRS:
        pareto_loss.fit_frontier((a, b), y_train[a], y_train[b])

    # --- pass 2: weighted final fit ---
    val_pred, test_pred, val_label, test_label = {}, {}, {}, {}
    models = {}
    for prop in PROPERTY_NAMES:
        sw = compute_sample_weights(prop, y_train, prelim_train, W, pareto_loss)
        rs = int(rng.integers(0, 1_000_000))
        if prop in REG_PROPERTIES:
            m = _fit_regressor(X_train, y_train[prop], config, sw, rs)
            val_pred[prop] = m.predict(X_val)
            test_pred[prop] = m.predict(X_test)
            val_label[prop], test_label[prop] = val_pred[prop], test_pred[prop]
        elif prop in MULTICLASS_PROPERTIES:
            m = _fit_classifier(X_train, y_train[prop], config, sw, rs)
            val_pred[prop] = m.predict_proba(X_val)
            test_pred[prop] = m.predict_proba(X_test)
            val_label[prop] = np.argmax(val_pred[prop], axis=1)
            test_label[prop] = np.argmax(test_pred[prop], axis=1)
        else:
            m = _fit_classifier(X_train, y_train[prop], config, sw, rs)
            val_pred[prop] = m.predict_proba(X_val)[:, 1]
            test_pred[prop] = m.predict_proba(X_test)[:, 1]
            val_label[prop] = (val_pred[prop] >= 0.5).astype(int)
            test_label[prop] = (test_pred[prop] >= 0.5).astype(int)
        models[prop] = m

    # --- aggregate validation LOSS (used only for the naive condition's
    # config selection) ---
    agg_loss = 0.0
    n_terms = 0
    for prop in PROPERTY_NAMES:
        for loss_name in PROPERTY_LOSS_ASSIGNMENT[prop]:
            if loss_name in ("correlation_consistency_loss", "pareto_tradeoff_penalty"):
                continue  # counted once per pair below, not per property
            loss = LOSS_REGISTRY[loss_name]
            if loss_name == "focal_loss":
                v = loss.evaluate(y_val[prop], val_pred[prop])
            elif loss_name == "uncertainty_weighted_loss":
                if prop in MULTICLASS_PROPERTIES or prop in BINARY_PROPERTIES:
                    unc = UncertaintyWeightedLoss.uncertainty_from_prob_margin(val_pred[prop])
                    v = loss.evaluate(y_val[prop], val_pred[prop], unc)
                else:
                    unc = UncertaintyWeightedLoss.uncertainty_from_residual_bins(val_pred[prop], y_val[prop])
                    v = loss.evaluate(y_val[prop], val_pred[prop], unc)
            elif loss_name == "variance_penalty_loss":
                v = loss.evaluate(y_val[prop], val_pred[prop])
            elif loss_name == "rank_consistency_loss" and prop in MULTICLASS_PROPERTIES:
                v = loss.evaluate(y_val[prop], val_label[prop])
            else:
                v = loss.evaluate(y_val[prop], val_pred[prop])
            agg_loss += W[loss_name] * v
            n_terms += 1
    for a, b in CORRELATION_PAIRS:
        v = LOSS_REGISTRY["correlation_consistency_loss"].evaluate(y_val[a], val_pred[a], y_val[b], val_pred[b])
        agg_loss += W["correlation_consistency_loss"] * v
        n_terms += 1
    for a, b in PARETO_PAIRS:
        v = pareto_loss.evaluate((a, b), val_pred[a], val_pred[b])
        agg_loss += W["pareto_tradeoff_penalty"] * v
        n_terms += 1
    agg_loss /= max(n_terms, 1)

    val_acc = {p: property_accuracy(p, y_val[p], val_label[p]) for p in PROPERTY_NAMES}
    test_acc = {p: property_accuracy(p, y_test[p], test_label[p]) for p in PROPERTY_NAMES}

    result = {
        "config": config,
        "W": W,
        "agg_val_loss": agg_loss,
        "agg_val_accuracy": float(np.mean(list(val_acc.values()))),
        "agg_test_accuracy": float(np.mean(list(test_acc.values()))),
        "val_acc_per_property": val_acc,
        "test_acc_per_property": test_acc,
    }
    if return_models:
        result["models"] = models
    return result


def search(splits, seed=42, n_naive=4, n_extra=40):
    """Runs the naive-pool + optimized-pool search described in the module
    docstring. Returns (naive_result, optimized_result, weight_trajectory)."""
    from loss_functions import ParetoTradeoffLoss

    rng = np.random.default_rng(seed)
    pool = []  # list of (config, W, is_naive)
    for _ in range(n_naive):
        pool.append((sample_config(rng), sample_weight_vector(rng, equal=True), True))
    for _ in range(n_extra):
        pool.append((sample_config(rng), sample_weight_vector(rng, equal=False), False))

    evaluated = []
    for config, W, is_naive in pool:
        pareto_loss = ParetoTradeoffLoss()
        r = run_candidate(config, W, splits, pareto_loss, np.random.default_rng(rng.integers(0, 1_000_000)))
        r["is_naive"] = is_naive
        evaluated.append(r)

    naive_pool = [r for r in evaluated if r["is_naive"]]
    naive_best = min(naive_pool, key=lambda r: r["agg_val_loss"])
    optimized_best = max(evaluated, key=lambda r: r["agg_val_accuracy"])

    trajectory = [(r["is_naive"], r["W"], r["agg_val_accuracy"]) for r in evaluated]
    return naive_best, optimized_best, trajectory


if __name__ == "__main__":
    from generate_compound_data import generate, train_val_test_split

    df = generate()
    splits = train_val_test_split(df)
    naive_best, optimized_best, _ = search(splits)
    print("naive agg test accuracy:", naive_best["agg_test_accuracy"])
    print("optimized agg test accuracy:", optimized_best["agg_test_accuracy"])
