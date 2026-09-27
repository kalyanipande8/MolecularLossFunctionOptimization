"""
8 custom, hand-implemented multi-objective loss functions for molecular
property prediction, applied across 12 properties (see generate_compound_data.py).

Because this machine has no PyTorch (see README for why) and critically low
disk space, there is no autograd. Each loss function here instead exposes
TWO things, which together form the "training objective" used by
train_multi_objective.py:

  1. `.evaluate(y_true, y_pred, **kw) -> float`
     A real scalar loss value, used for evaluation/reporting and for scoring
     candidate models during weight search.

  2. `.sample_weights(y_true, y_pred_prelim, **kw) -> np.ndarray`
     A per-sample training weight signal derived directly from the loss's
     own mathematical form (e.g. focal loss's (1-p)^gamma term, Huber's IRLS
     re-descending weight, inverse-variance weighting for the uncertainty
     loss, etc). This is a standard, real technique for approximating a
     custom loss's effect on a model that is trained via `sample_weight`
     (scikit-learn regressors/classifiers all accept `sample_weight`)
     rather than by direct backprop on the loss itself.

A per-property model is trained in two passes (see train_multi_objective.py):
first an unweighted pass to get preliminary predictions/residuals, then a
second pass using `sample_weights` computed from those preliminary residuals
and the 8 losses assigned to that property. The GLOBAL weight vector `W`
(one scalar per loss function, 8 numbers total) scales how strongly each
loss's sample-reweighting signal is mixed in; W is exactly what
train_multi_objective.py's naive-vs-optimized search tunes.
"""
from dataclasses import dataclass
import numpy as np


def _safe_std(x, floor=1e-3):
    return max(float(np.std(x)), floor)


# ---------------------------------------------------------------------------
# 1. Huber loss -- robust regression loss for noisy continuous targets
# ---------------------------------------------------------------------------
class HuberLoss:
    """Robust regression loss: quadratic near zero, linear in the tails, so a
    handful of noisy/outlier compounds don't dominate training. Applied to
    solubility, logp, clearance (all noisy, no long tail of "hard" cases)."""

    name = "huber_loss"

    def __init__(self, delta=1.0):
        self.delta = delta

    def evaluate(self, y_true, y_pred):
        r = np.asarray(y_true) - np.asarray(y_pred)
        abs_r = np.abs(r)
        quad = 0.5 * r ** 2
        lin = self.delta * (abs_r - 0.5 * self.delta)
        return float(np.mean(np.where(abs_r <= self.delta, quad, lin)))

    def sample_weights(self, y_true, y_pred_prelim):
        r = np.asarray(y_true) - np.asarray(y_pred_prelim)
        abs_r = np.abs(r) + 1e-8
        # IRLS re-descending weight: full weight inside delta, downweighted
        # (1/|r|) outside it -- outliers get less influence on the refit.
        return np.where(abs_r <= self.delta, 1.0, self.delta / abs_r)


# ---------------------------------------------------------------------------
# 2. Focal loss -- for imbalanced binary classification targets
# ---------------------------------------------------------------------------
class FocalLoss:
    """Down-weights easy, already-correctly-classified majority-class
    examples and focuses training on hard/minority examples. Applied to
    toxicity_class and herg_liability (both ~15-20% positive rate)."""

    name = "focal_loss"

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def evaluate(self, y_true, y_prob):
        y_true = np.asarray(y_true)
        p = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        loss = -alpha_t * (1 - p_t) ** self.gamma * np.log(p_t)
        return float(np.mean(loss))

    def sample_weights(self, y_true, y_prob_prelim):
        y_true = np.asarray(y_true)
        p = np.clip(np.asarray(y_prob_prelim), 1e-6, 1 - 1e-6)
        p_t = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        return alpha_t * (1 - p_t) ** self.gamma


# ---------------------------------------------------------------------------
# 3. Rank-consistency loss -- for ordinal / rank-meaningful properties
# ---------------------------------------------------------------------------
class RankConsistencyLoss:
    """Penalizes pairwise order violations: if compound i truly has a higher
    value than compound j, the model should predict it higher too. Better
    suited to properties whose *ordering* across compounds matters (e.g. for
    prioritization/triage) than their exact value. Applied to melting_point,
    synthetic_accessibility, binding_class (an ordinal 3-class potency
    bucket)."""

    name = "rank_consistency_loss"

    def __init__(self, n_pairs=4000, seed=0):
        self.n_pairs = n_pairs
        self.seed = seed

    def _pairs(self, n):
        rng = np.random.default_rng(self.seed)
        n_pairs = min(self.n_pairs, n * (n - 1) // 2)
        i = rng.integers(0, n, n_pairs)
        j = rng.integers(0, n, n_pairs)
        mask = i != j
        return i[mask], j[mask]

    def evaluate(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        i, j = self._pairs(len(y_true))
        true_sign = np.sign(y_true[i] - y_true[j])
        pred_diff = y_pred[i] - y_pred[j]
        # hinge-style pairwise ranking loss: margin violation when order flips
        margin = 0.1
        viol = np.maximum(0.0, margin - true_sign * pred_diff)
        viol = viol[true_sign != 0]
        return float(np.mean(viol)) if len(viol) else 0.0

    def sample_weights(self, y_true, y_pred_prelim):
        y_true = np.asarray(y_true, dtype=float)
        y_pred_prelim = np.asarray(y_pred_prelim, dtype=float)
        n = len(y_true)
        i, j = self._pairs(n)
        true_sign = np.sign(y_true[i] - y_true[j])
        pred_diff = y_pred_prelim[i] - y_pred_prelim[j]
        discordant = (true_sign * pred_diff < 0) & (true_sign != 0)
        # weight of a sample = how often it participates in a discordant pair
        w = np.ones(n)
        counts = np.zeros(n)
        idx = np.concatenate([i[discordant], j[discordant]])
        np.add.at(counts, idx, 1.0)
        if counts.max() > 0:
            w += counts / counts.max()
        return w


# ---------------------------------------------------------------------------
# 4. Property-correlation-consistency regularizer
# ---------------------------------------------------------------------------
class CorrelationConsistencyLoss:
    """Regularizes a *pair* of positively-correlated properties so the model
    doesn't predict a relationship between them that contradicts the real
    data (e.g. predicting melting_point up but synthetic_accessibility down
    when in the real data they move together via a shared rigidity factor).
    Applied to (melting_point, synthetic_accessibility) and
    (logp, protein_binding) -- both pairs share an underlying chemistry
    driver in the synthetic data-generating process."""

    name = "correlation_consistency_loss"

    def evaluate(self, y_true_a, y_pred_a, y_true_b, y_pred_b):
        true_corr = np.corrcoef(y_true_a, y_true_b)[0, 1]
        pred_corr = np.corrcoef(y_pred_a, y_pred_b)[0, 1]
        if np.isnan(true_corr) or np.isnan(pred_corr):
            return 0.0
        return float((true_corr - pred_corr) ** 2)

    def sample_weights(self, y_true_a, y_pred_prelim_a, y_true_b, y_pred_prelim_b):
        # weight up samples whose (a,b) residual pattern disagrees with the
        # population-level correlation sign (they are the ones dragging
        # predicted correlation away from the true relationship)
        true_corr = np.corrcoef(y_true_a, y_true_b)[0, 1]
        ra = y_true_a - y_pred_prelim_a
        rb = y_true_b - y_pred_prelim_b
        agree = np.sign(ra * rb) == np.sign(true_corr if true_corr != 0 else 1.0)
        w = np.ones(len(y_true_a))
        w[~agree] = 1.5
        return w


# ---------------------------------------------------------------------------
# 5. Variance-penalty loss -- discourage overconfidence on inherently-hard
#    properties
# ---------------------------------------------------------------------------
class VariancePenaltyLoss:
    """Penalizes the model when its squared error on a sample is wildly out
    of line with the locally-estimated (binned) noise variance for that
    region of property space -- i.e. discourages overconfident predictions
    on compounds that are genuinely hard to predict. Applied to
    binding_affinity, which has heteroscedastic (compound-dependent) noise
    baked into the synthetic data generator."""

    name = "variance_penalty_loss"

    def __init__(self, n_bins=8):
        self.n_bins = n_bins

    def _local_var(self, y_true, y_pred_prelim):
        r2 = (np.asarray(y_true) - np.asarray(y_pred_prelim)) ** 2
        order = np.argsort(y_pred_prelim)
        bins = np.array_split(order, self.n_bins)
        local_var = np.zeros(len(y_true))
        for b in bins:
            local_var[b] = max(np.mean(r2[b]), 1e-3)
        return local_var

    def evaluate(self, y_true, y_pred, y_pred_prelim=None):
        y_pred_prelim = y_pred if y_pred_prelim is None else y_pred_prelim
        local_var = self._local_var(y_true, y_pred_prelim)
        r2 = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
        return float(np.mean((r2 - local_var) ** 2))

    def sample_weights(self, y_true, y_pred_prelim):
        local_var = self._local_var(y_true, y_pred_prelim)
        # downweight high-noise-region samples so the model doesn't chase
        # unlearnable noise there
        return 1.0 / (1.0 + local_var / np.mean(local_var))


# ---------------------------------------------------------------------------
# 6. Tolerance-band loss -- for properties with an acceptable RANGE, not an
#    exact target value
# ---------------------------------------------------------------------------
class ToleranceBandLoss:
    """Zero loss inside an acceptable band around the true value (a real
    ADMET-style property is often "fine" anywhere in a developability
    window, not just at one exact number); loss grows outside the band.
    Applied to permeability and protein_binding, which in real drug
    discovery are judged against acceptable ranges rather than exact
    values."""

    name = "tolerance_band_loss"

    def __init__(self, band=0.5):
        self.band = band

    def evaluate(self, y_true, y_pred):
        d = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        excess = np.maximum(0.0, d - self.band)
        return float(np.mean(excess ** 2))

    def sample_weights(self, y_true, y_pred_prelim):
        d = np.abs(np.asarray(y_true) - np.asarray(y_pred_prelim))
        outside = d > self.band
        w = np.ones(len(y_true))
        w[outside] = 1.0 + (d[outside] - self.band)
        return w


# ---------------------------------------------------------------------------
# 7. Uncertainty-weighted loss -- inverse-variance weighted objective
# ---------------------------------------------------------------------------
class UncertaintyWeightedLoss:
    """Heteroscedastic-style loss: samples with higher estimated predictive
    uncertainty contribute less to the objective (inverse-variance
    weighting), a standard technique for multi-task/multi-property learning
    when some properties/samples are noisier than others. Applied to
    binding_affinity (paired with the variance-penalty loss) and
    metabolic_stability_class (uncertainty proxied by prediction margin)."""

    name = "uncertainty_weighted_loss"

    def evaluate(self, y_true, y_pred, uncertainty):
        r2 = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
        u = np.clip(np.asarray(uncertainty), 1e-3, None)
        return float(np.mean(r2 / (2 * u ** 2) + np.log(u)))

    def sample_weights(self, uncertainty):
        u = np.clip(np.asarray(uncertainty), 1e-3, None)
        return 1.0 / (u ** 2)

    @staticmethod
    def uncertainty_from_residual_bins(y_pred_prelim, y_true, n_bins=8):
        r = np.abs(np.asarray(y_true) - np.asarray(y_pred_prelim))
        order = np.argsort(y_pred_prelim)
        bins = np.array_split(order, n_bins)
        u = np.zeros(len(y_true))
        for b in bins:
            u[b] = max(np.std(r[b]), 1e-2)
        return u

    @staticmethod
    def uncertainty_from_prob_margin(y_prob):
        # distance from a confident decision (0 or 1) -> larger near 0.5
        return 0.5 - np.abs(np.asarray(y_prob) - 0.5) + 1e-2


# ---------------------------------------------------------------------------
# 8. Pareto / trade-off-aware penalty
# ---------------------------------------------------------------------------
class ParetoTradeoffLoss:
    """For two properties known to trade off (improving one tends to hurt
    the other, via a shared latent chemistry factor), this penalizes
    predictions that violate the empirical Pareto frontier estimated from
    training data more than the real compounds do -- i.e. it discourages the
    model from predicting an unrealistic "free lunch" (both properties
    great at once) that isn't supported by the data. Applied to the
    trade-off pairs (solubility, logp) and (permeability, clearance)."""

    name = "pareto_tradeoff_penalty"

    def __init__(self):
        self._slope = {}
        self._resid_std = {}

    def fit_frontier(self, key, true_a, true_b):
        slope = np.polyfit(true_a, true_b, 1)[0]
        resid = true_b - slope * true_a
        self._slope[key] = slope
        self._resid_std[key] = _safe_std(resid)

    def evaluate(self, key, pred_a, pred_b):
        slope = self._slope[key]
        resid = pred_b - slope * pred_a
        resid_std = self._resid_std[key]
        # penalize only the "overshoot" direction: predicted residual much
        # better than the real trade-off frontier allows (unrealistic),
        # scaled by how many real-data std devs beyond the frontier it is.
        overshoot = np.maximum(0.0, np.abs(resid) - 2 * resid_std)
        return float(np.mean((overshoot / resid_std) ** 2))

    def sample_weights(self, key, true_a, pred_a_prelim, true_b, pred_b_prelim):
        slope = self._slope[key]
        resid_std = self._resid_std[key]
        resid = pred_b_prelim - slope * pred_a_prelim
        overshoot = np.abs(resid) > 2 * resid_std
        w = np.ones(len(true_a))
        w[overshoot] = 1.5
        return w


# ---------------------------------------------------------------------------
# Assignment table: which of the 8 losses applies to which of the 12
# properties. Documented rationale is in each class's docstring above.
# ---------------------------------------------------------------------------
LOSS_REGISTRY = {
    "huber_loss": HuberLoss(),
    "focal_loss": FocalLoss(),
    "rank_consistency_loss": RankConsistencyLoss(),
    "correlation_consistency_loss": CorrelationConsistencyLoss(),
    "variance_penalty_loss": VariancePenaltyLoss(),
    "tolerance_band_loss": ToleranceBandLoss(),
    "uncertainty_weighted_loss": UncertaintyWeightedLoss(),
    "pareto_tradeoff_penalty": ParetoTradeoffLoss(),
}

# property -> list of loss names applied to it
PROPERTY_LOSS_ASSIGNMENT = {
    "solubility": ["huber_loss", "pareto_tradeoff_penalty"],
    "logp": ["huber_loss", "pareto_tradeoff_penalty", "correlation_consistency_loss"],
    "permeability": ["tolerance_band_loss", "pareto_tradeoff_penalty"],
    "clearance": ["huber_loss", "pareto_tradeoff_penalty"],
    "protein_binding": ["tolerance_band_loss", "correlation_consistency_loss"],
    "melting_point": ["rank_consistency_loss", "correlation_consistency_loss"],
    "binding_affinity": ["variance_penalty_loss", "uncertainty_weighted_loss"],
    "synthetic_accessibility": ["rank_consistency_loss", "correlation_consistency_loss"],
    "toxicity_class": ["focal_loss"],
    "herg_liability": ["focal_loss"],
    "binding_class": ["rank_consistency_loss"],
    "metabolic_stability_class": ["uncertainty_weighted_loss"],
}

# cross-property pairs used by the two pairwise losses
CORRELATION_PAIRS = [("melting_point", "synthetic_accessibility"), ("logp", "protein_binding")]
PARETO_PAIRS = [("solubility", "logp"), ("permeability", "clearance")]

ALL_LOSS_NAMES = list(LOSS_REGISTRY.keys())
