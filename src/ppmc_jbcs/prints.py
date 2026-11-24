import pandas as pd

from .statics_metrics import compute_properties, print_comparison


def print_datasets(df, X_anon, feature_columns):
    """Print the first rows of the original and anonymized datasets."""
    print("Original data:")
    print(df.head(5))

    print("\nAnonymized data:")
    print(pd.DataFrame(X_anon, columns=feature_columns).head(5))


def print_preserved_properties(X, X_anon):
    """Print each preserved property for the original and anonymized data."""
    original_properties = compute_properties(X)
    anonymized_properties = compute_properties(X_anon)
    print("\nPreserved properties (original vs anonymized):")
    for name in original_properties:
        max_items = None if name in ("Mean", "Eigenvalue Spectrum") else 5
        print_comparison(
            name,
            original_properties[name],
            anonymized_properties[name],
            max_items=max_items,
        )
