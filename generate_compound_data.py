"""
Generate synthetic "compound" data for multi-objective molecular property
loss function optimization.

Not real molecules / real assay data. Each of the 560 synthetic compounds
gets a feature vector standing in for molecular descriptors (things like
molecular-weight-like, polar-surface-area-like, lipophilicity-like scalar
descriptors) and 12 distinct molecular property targets, generated from a
shared latent "chemistry" so that several pairs of properties are genuinely
in tension (improving one degrades another), which is what makes multi-
objective loss weighting matter instead of being cosmetic.

Fixed random seed everywhere -> fully reproducible.
"""
import numpy as np
import pandas as pd

SEED = 42
N_COMPOUNDS = 560
N_FEATURES = 20

# 12 molecular properties. 8 regression-style ("reg"), 4 classification-style
# ("clf"), mixing continuous-valued, tolerance-band, and ordinal-like targets.
PROPERTY_NAMES = [
    "solubility",          # reg  - continuous, noisy
    "logp",                # reg  - continuous, noisy, in tension with solubility
    "permeability",        # reg  - continuous, tolerance-band ("acceptable range")
    "clearance",            # reg  - continuous, noisy, in tension with permeability
    "protein_binding",      # reg  - continuous, tolerance-band
    "melting_point",        # reg  - continuous, ordinal-ish (bucketed also used for ranking)
    "binding_affinity",     # reg  - continuous, hard/noisy tail (variance-penalty candidate)
    "synthetic_accessibility",  # reg - continuous, ordinal (rank-consistency candidate)
    "toxicity_class",       # clf  - binary, imbalanced (~18% positive)
    "herg_liability",       # clf  - binary, imbalanced (~15% positive), in tension with logp
    "binding_class",        # clf  - multi-class (3 classes) ordinal-ish potency bucket
    "metabolic_stability_class",  # clf - binary, moderately balanced
]

REG_PROPERTIES = PROPERTY_NAMES[:8]
CLF_PROPERTIES = PROPERTY_NAMES[8:]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n=N_COMPOUNDS, n_features=N_FEATURES, seed=SEED):
    rng = np.random.default_rng(seed)

    # Latent "chemistry" factors that drive several properties at once so
    # that trade-offs between properties are real, not just noise.
    lipophilicity_factor = rng.normal(0, 1, n)   # drives logp up, solubility down
    size_factor = rng.normal(0, 1, n)            # drives MW-like effects
    polarity_factor = rng.normal(0, 1, n)        # drives solubility up, permeability down
    rigidity_factor = rng.normal(0, 1, n)        # drives synthetic accessibility / stability
    electronic_factor = rng.normal(0, 1, n)      # drives binding affinity / toxicity

    # Feature matrix: noisy linear mixtures of the latent factors plus
    # independent noise columns, standing in for 20 molecular descriptors.
    latent = np.stack(
        [lipophilicity_factor, size_factor, polarity_factor, rigidity_factor, electronic_factor],
        axis=1,
    )
    mixing = rng.normal(0, 1, size=(5, n_features))
    X = latent @ mixing + rng.normal(0, 0.6, size=(n, n_features))

    props = {}

    # --- Regression targets -------------------------------------------------
    # solubility: favored by polarity, hurt by lipophilicity -> tension w/ logp
    props["solubility"] = (
        2.0 * polarity_factor - 1.8 * lipophilicity_factor + rng.normal(0, 0.7, n)
    )
    # logp: favored by lipophilicity, hurt by polarity (direct trade-off pair)
    props["logp"] = (
        1.9 * lipophilicity_factor - 1.2 * polarity_factor + rng.normal(0, 0.5, n)
    )
    # permeability: favored by lipophilicity, hurt by polarity & size
    # tolerance-band property: only whether it's within an acceptable window matters
    props["permeability"] = (
        1.5 * lipophilicity_factor - 1.0 * polarity_factor - 0.6 * size_factor
        + rng.normal(0, 0.8, n)
    )
    # clearance: favored by small size + low lipophilicity; in tension with permeability
    props["clearance"] = (
        -1.3 * lipophilicity_factor + 0.9 * size_factor + rng.normal(0, 0.9, n)
    )
    # protein_binding: driven by lipophilicity + electronics; tolerance-band property
    props["protein_binding"] = (
        1.1 * lipophilicity_factor + 0.8 * electronic_factor + rng.normal(0, 0.6, n)
    )
    # melting_point: driven by rigidity + size, fairly ordinal/bucketable
    props["melting_point"] = (
        1.7 * rigidity_factor + 0.7 * size_factor + rng.normal(0, 0.6, n)
    )
    # binding_affinity: driven by electronics, with a noisy hard-to-predict tail
    # (heteroscedastic noise -> genuinely hard for some compounds, good target
    # for a variance-penalty loss)
    hardness = _sigmoid(1.2 * rigidity_factor)  # some compounds are just noisier
    props["binding_affinity"] = (
        1.6 * electronic_factor + rng.normal(0, 0.4 + 1.3 * hardness, n)
    )
    # synthetic_accessibility: driven by rigidity + size, monotonic/ordinal in
    # nature (easier vs harder to synthesize), good rank-consistency target
    props["synthetic_accessibility"] = (
        1.3 * rigidity_factor + 0.5 * size_factor + rng.normal(0, 0.5, n)
    )

    # --- Classification targets ---------------------------------------------
    # toxicity_class: imbalanced binary, driven by electronics + lipophilicity
    tox_logit = 1.4 * electronic_factor + 0.9 * lipophilicity_factor - 1.1
    props["toxicity_class"] = (rng.uniform(0, 1, n) < _sigmoid(tox_logit)).astype(int)

    # herg_liability: imbalanced binary, strongly driven by lipophilicity ->
    # directly in tension with logp/permeability being pushed up
    herg_logit = 1.6 * lipophilicity_factor - 1.3
    props["herg_liability"] = (rng.uniform(0, 1, n) < _sigmoid(herg_logit)).astype(int)

    # binding_class: 3-class ordinal potency bucket (weak/medium/strong),
    # derived from binding_affinity, so classes have a genuine order
    ba = props["binding_affinity"]
    q1, q2 = np.quantile(ba, [1 / 3, 2 / 3])
    props["binding_class"] = np.digitize(ba, [q1, q2])  # 0,1,2

    # metabolic_stability_class: moderately balanced binary, driven by
    # clearance (low clearance -> stable)
    stab_logit = -1.1 * props["clearance"] + 0.2
    props["metabolic_stability_class"] = (rng.uniform(0, 1, n) < _sigmoid(stab_logit)).astype(int)

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    for name in PROPERTY_NAMES:
        df[name] = props[name]

    # standardize regression targets to roughly unit scale for stable training
    for name in REG_PROPERTIES:
        mu, sd = df[name].mean(), df[name].std()
        df[name] = (df[name] - mu) / sd

    return df


def train_val_test_split(df, seed=SEED, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(train_frac * len(df))
    n_val = int(val_frac * len(df))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df.iloc[train_idx].reset_index(drop=True), \
        df.iloc[val_idx].reset_index(drop=True), \
        df.iloc[test_idx].reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    print(f"Generated {len(df)} compounds x {len(PROPERTY_NAMES)} properties")
    print(df[PROPERTY_NAMES].describe().T[["mean", "std", "min", "max"]])
    tr, va, te = train_val_test_split(df)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    df.to_csv("output/compounds.csv", index=False)
