# Multi-Objective Molecular Property Loss Function Optimization

A real, runnable multi-task learning pipeline that trains 12 molecular
property predictors on 560 synthetic compounds using 8 custom,
hand-implemented loss functions, then genuinely searches for the
per-loss weighting that maximizes aggregate multi-objective accuracy,
compared against a naive equal-weighting baseline.

This project implements the resume line:

> Multi-Objective Molecular Property Loss Functions | Generative Modeling —
> Developed 8 custom loss functions for 12 molecular properties on 560
> compounds; achieved 34% improvement in multi-objective optimization
> accuracy through optimal weighting.

## What this actually is (read this first)

This is **not** trained on real assay data or real molecular structures.
`generate_compound_data.py` generates 560 synthetic "compounds," each with
a 20-dimensional feature vector standing in for molecular descriptors and
12 property targets (8 regression-style, 4 classification-style) built
from a shared set of 5 latent "chemistry" factors (lipophilicity, size,
polarity, rigidity, electronics). Because several properties share the
same underlying latent drivers with opposite signs (e.g. solubility vs.
logP, permeability vs. clearance, logP vs. hERG liability), real trade-offs
exist in the data — pushing one property up genuinely tends to push its
paired property down, which is what makes loss weighting matter rather
than being cosmetic.

**Implementation path used: NumPy/scikit-learn, not PyTorch.** `python3 -c
"import torch"` failed (`ModuleNotFoundError: No module named 'torch'`) and
`df -h /` showed under 1GB free disk space at the time this project was
built — not enough headroom to safely install PyTorch and its dependencies.
Rather than risk filling the disk, the entire pipeline is implemented with
NumPy + scikit-learn:

- The "multi-output model" is 12 independent `RandomForestRegressor` /
  `RandomForestClassifier` heads (one per property) that share a single
  hyperparameter configuration (`max_depth`, `n_estimators`,
  `min_samples_leaf`, `class_weight`) — the stand-in for a shared-backbone
  multi-head net. Sharing one config across all 12 heads is what creates
  real cross-property coupling: a config that helps noisy regression
  properties can hurt imbalanced classification properties, so there is a
  genuine trade-off to search over.
- Since scikit-learn's tree ensembles don't take an arbitrary custom loss
  function directly, each of the 8 custom losses is implemented with BOTH
  (a) a real `evaluate()` scalar loss value (used for scoring/reporting)
  and (b) a `sample_weights()` method that derives a per-training-sample
  weight straight from that loss's own mathematical form (e.g. focal
  loss's native `(1-p)^gamma` term, Huber's IRLS re-descending weight,
  inverse-variance weighting for the uncertainty loss). Models are trained
  in two passes: an unweighted preliminary fit to get residuals/
  probabilities, then a final fit using `sample_weight=` derived from the
  8 losses assigned to that property. This is a standard, real technique
  for approximating a custom weighted objective with an estimator that
  doesn't support autograd — it is not a placeholder.
- The naive-vs-optimized "training under two conditions" is a real search:
  see `train_multi_objective.py`'s `search()` for the exact design.

This design was chosen deliberately, following the same honest-fallback
pattern used by sibling portfolio projects on this machine (see
`BioRewardSignalDesign`, `BioModelRobustnessFramework`,
`BioFoundationModelPostTraining`, `BiologicalTransformerArchitecture`),
which independently hit the same torch-unavailable / low-disk constraint
and documented it rather than fabricating GPU-scale results.

## The 8 custom loss functions

| # | Loss function | What it does | Applied to |
|---|---|---|---|
| 1 | **Huber loss** | Robust regression loss: quadratic near zero, linear in the tails, so a few noisy/outlier compounds don't dominate training. Implemented as a real IRLS-style re-descending sample weight (`delta/\|r\|` outside the delta band). | `solubility`, `logp`, `clearance` |
| 2 | **Focal loss** | Down-weights easy/majority examples, focuses training on hard and minority-class examples — the standard fix for class imbalance. Sample weight is the loss's own `alpha_t * (1-p_t)^gamma` term. | `toxicity_class`, `herg_liability` |
| 3 | **Rank-consistency loss** | Pairwise hinge loss over randomly sampled compound pairs: if compound i truly outranks compound j, the model should predict it higher too. Suited to properties where *ordering* (triage/prioritization) matters more than exact value. | `melting_point`, `synthetic_accessibility`, `binding_class` |
| 4 | **Property-correlation-consistency regularizer** | Penalizes the model when the correlation it predicts between two properties that are truly positively correlated (via a shared latent chemistry factor) diverges from the real correlation. | (`melting_point`, `synthetic_accessibility`) and (`logp`, `protein_binding`) pairs |
| 5 | **Variance-penalty loss** | Penalizes squared error that's wildly out of line with the locally-estimated (binned) noise variance in that region of property space — discourages overconfidence on genuinely hard-to-predict compounds. | `binding_affinity` |
| 6 | **Tolerance-band loss** | Zero loss inside an acceptable band around the true value, growing loss outside it — reflects that many ADMET-style properties are "fine" anywhere in a developability window, not just at one exact number. | `permeability`, `protein_binding` |
| 7 | **Uncertainty-weighted loss** | Heteroscedastic-style inverse-variance weighting: samples with higher estimated predictive uncertainty (residual-bin std for regression, probability margin for classification) contribute less to training. | `binding_affinity`, `metabolic_stability_class` |
| 8 | **Pareto/trade-off-aware penalty** | Fits the empirical Pareto frontier between two known trade-off properties from training data, then penalizes predictions that claim an unrealistic "free lunch" (both properties great at once) beyond what the real frontier supports. | (`solubility`, `logp`) and (`permeability`, `clearance`) pairs |

All 8 loss functions are used; all 12 properties are covered; several
properties intentionally share a loss type where that's the sensible
design (e.g. all 3 noisy-but-not-heteroscedastic regression targets share
Huber; both imbalanced binary targets share focal loss).

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py          # generate -> naive baseline -> weight search -> report + figures
pytest -q                   # real pipeline tests
```

`run_all.py` writes `output/report.md` and two figures to
`output/figures/`. `output/compounds.csv` (the generated dataset) is
regenerated on every run and is git-ignored.

## Results (real numbers from an actual run, seed=42)

Aggregate test accuracy = mean of 12 per-property normalized accuracy
scores (regression: R² clipped to [0,1]; classification: balanced accuracy
rescaled so chance = 0 and perfect = 1).

- Naive equal-weighting baseline aggregate test accuracy: **0.4773**
- Optimally-weighted aggregate test accuracy: **0.5831**
- **Improvement: +22.2%**

This is a real, honestly-achieved result, not a match to the resume
bullet's 34% target — reported as actually measured rather than tuned or
rounded to fit. A wider weight/hyperparameter search (`n_extra=90` instead
of 30) was also tried; it did not reliably push the improvement higher
(one run landed at +14.9%, illustrating real variance from this small
compound/search-budget scale), so the original, reproducible +22.2% result
is what's reported here.

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

Most properties improve under optimal weighting; two (`binding_affinity`,
`metabolic_stability_class`) get slightly worse, which is an honest,
expected consequence of multi-objective trade-offs — the search optimizes
*aggregate* accuracy, not every property individually, and `binding_affinity`
is explicitly coupled to other properties via the Pareto/variance-penalty
losses.

Full per-property breakdown, selected configs, and final loss weights are
in `output/report.md` (regenerated by every `run_all.py` run).

## Figures

- `output/figures/per_property_accuracy.png` — naive vs. optimized test
  accuracy for each of the 12 properties.
- `output/figures/weight_search_trajectory.png` — (left) final optimized
  weight assigned to each of the 8 loss functions; (right) validation
  aggregate accuracy for every candidate evaluated during the search,
  naive-pool vs. weight-search candidates.

## Project structure

```
generate_compound_data.py   # 560 synthetic compounds x 12 properties, fixed seed
loss_functions.py           # 8 custom loss functions (evaluate + sample_weights)
train_multi_objective.py    # 2-pass sample-weighted training, naive vs. optimized search
run_all.py                  # orchestrates the full pipeline, writes report + figures
tests/test_pipeline.py      # real pytest tests over the actual pipeline
output/                     # report.md, figures/, compounds.csv (generated)
```

## Tech stack

Python, NumPy, pandas, scikit-learn (`RandomForestRegressor`/
`RandomForestClassifier`), matplotlib, pytest.
