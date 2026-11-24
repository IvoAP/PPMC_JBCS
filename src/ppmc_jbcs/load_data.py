from pathlib import Path

import kagglehub
import pandas as pd


def load_heart_disease() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download("johnsmith88/heart-disease-dataset"))
    csv_files = sorted(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {dataset_path}")
    return pd.read_csv(csv_files[0])
