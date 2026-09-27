# Multi-Objective Molecular Property Loss Optimization -- Results

Aggregate test accuracy (mean of 12 normalized per-property scores):

- Naive equal-weighting baseline: **0.4773**
- Optimally-weighted model:       **0.5831**
- Improvement:                    **+22.2%**

## Per-property test accuracy (naive vs optimized)

| Property | Naive | Optimized | Delta |
|---|---|---|---|
| solubility | 0.719 | 0.853 | +0.133 |
| logp | 0.650 | 0.820 | +0.170 |
| permeability | 0.670 | 0.765 | +0.095 |
| clearance | 0.616 | 0.662 | +0.046 |
| protein_binding | 0.496 | 0.544 | +0.049 |
| melting_point | 0.794 | 0.835 | +0.041 |
| binding_affinity | 0.351 | 0.152 | -0.199 |
| synthetic_accessibility | 0.743 | 0.819 | +0.076 |
| toxicity_class | 0.000 | 0.416 | +0.416 |
| herg_liability | 0.000 | 0.377 | +0.377 |
| binding_class | 0.291 | 0.364 | +0.073 |
| metabolic_stability_class | 0.396 | 0.388 | -0.008 |

## Selected configurations

- Naive config: `{'max_depth': 3, 'n_estimators': 50, 'min_samples_leaf': 2, 'class_weight': 'balanced'}`
- Naive loss weights (all equal): `{'huber_loss': 1.0, 'focal_loss': 1.0, 'rank_consistency_loss': 1.0, 'correlation_consistency_loss': 1.0, 'variance_penalty_loss': 1.0, 'tolerance_band_loss': 1.0, 'uncertainty_weighted_loss': 1.0, 'pareto_tradeoff_penalty': 1.0}`
- Optimized config: `{'max_depth': None, 'n_estimators': 50, 'min_samples_leaf': 4, 'class_weight': 'balanced'}`
- Optimized loss weights:
  - huber_loss: 0.742
  - focal_loss: 0.275
  - rank_consistency_loss: 1.797
  - correlation_consistency_loss: 3.970
  - variance_penalty_loss: 3.578
  - tolerance_band_loss: 3.020
  - uncertainty_weighted_loss: 3.574
  - pareto_tradeoff_penalty: 3.584
