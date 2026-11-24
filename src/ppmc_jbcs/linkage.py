import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .anon import RPAR

DEFAULT_KS = (1, 3, 5, 10)


def raw_space_scores(X, X_anon):
    """Distance between each anonymized record and each original, in standardized raw space.

    The attacker only knows the public feature statistics of the original data.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    return cdist((X_anon - mean) / std, (X - mean) / std)


def distance_profile_scores(X, X_anon):
    """Distance between the sorted distance profiles of anonymized and original records.

    The anonymization preserves all pairwise distances, so each record's profile is unchanged.
    """
    return cdist(_sorted_profiles(X_anon), _sorted_profiles(X))


def _sorted_profiles(data):
    """Return, for each record, its sorted distances to all other records."""
    distances = cdist(data, data)
    n_records = data.shape[0]
    off_diagonal = ~np.eye(n_records, dtype=bool)
    return np.sort(distances[off_diagonal].reshape(n_records, n_records - 1), axis=1)


ATTACKS = {
    "Raw-space matching": raw_space_scores,
    "Distance-profile matching": distance_profile_scores,
}


def compute_ranks(scores):
    """Return the rank (1 = best) of the true original for each anonymized record."""
    true_scores = np.diag(scores)
    return 1 + (scores < true_scores[:, None]).sum(axis=1)


def run_linkage(X, random_states, ks=DEFAULT_KS, perturb=None):
    """Run every attack over several anonymizations.

    ``perturb(X_anon, X, seed)``, if given, transforms each anonymized dataset before the attacks.
    Returns a DataFrame with the Top-k match rate per attack, seed and k, and a dict
    mapping each attack to the concatenated ranks of the true originals.
    """
    rows = []
    ranks = {name: [] for name in ATTACKS}
    for seed in random_states:
        X_anon = RPAR(random_state=seed).fit_transform(X)
        if perturb is not None:
            X_anon = perturb(X_anon, X, seed)
        for name, score_fn in ATTACKS.items():
            attack_ranks = compute_ranks(score_fn(X, X_anon))
            ranks[name].append(attack_ranks)
            for k in ks:
                rows.append({"attack": name, "seed": seed, "k": k, "match_rate": np.mean(attack_ranks <= k)})
    return pd.DataFrame(rows), {name: np.concatenate(values) for name, values in ranks.items()}
