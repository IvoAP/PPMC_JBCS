"""Central settings for the Clinical Privacy and Explainability Framework."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Resolve study inputs and outputs relative to a single workspace."""

    root: Path = Path(__file__).resolve().parent
    dataset_filename: str = "diabetic_data.csv"

    @property
    def dataset(self) -> Path:
        return self.root / "data" / "raw" / self.dataset_filename

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def directories(self) -> tuple[Path, ...]:
        data = self.root / "data"
        return (
            *(data / name for name in ("raw", "processed", "splits")),
            *(self.results / name for name in (
                "tables", "figures", "explanations", "models", "logs",
            )),
        )


@dataclass(frozen=True)
class Dataset:
    """Column roles and holdout proportions of the full dataset."""

    target: str = "readmitted"
    identifiers: tuple[str, ...] = ("encounter_id", "patient_nbr")
    test_fraction: float = 0.2
    validation_fraction: float = 0.1

    def __post_init__(self) -> None:
        if not (0 < self.test_fraction < 1 and 0 < self.validation_fraction < 1
                and self.test_fraction + self.validation_fraction < 1):
            raise ValueError("Split fractions must leave a nonempty training set.")
        if self.target in self.identifiers:
            raise ValueError("The target cannot be an identifier.")


@dataclass(frozen=True)
class Preprocessing:
    """Conservative feature allowlists by time of availability."""

    prediction_time: str = "admission"
    admission_numeric: tuple[str, ...] = (
        "number_outpatient", "number_emergency", "number_inpatient",
    )
    admission_categorical: tuple[str, ...] = (
        "age", "gender", "race", "admission_type_id", "admission_source_id",
    )
    discharge_numeric: tuple[str, ...] = (
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications",
    )
    discharge_categorical: tuple[str, ...] = ("max_glu_serum", "A1Cresult")
    missing_tokens: tuple[str, ...] = ("", "?")
    scale_numeric: bool = True

    def __post_init__(self) -> None:
        if self.prediction_time not in ("admission", "discharge"):
            raise ValueError("Prediction time must be admission or discharge.")


@dataclass(frozen=True)
class Anonymization:
    """Matched alpha sweep; clustered evaluation requires an explicit k."""

    alphas: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    clusters: int | None = None
    source_run: str | None = None
    categorical_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if self.clusters is not None and self.clusters < 2:
            raise ValueError("Cluster count must be at least two.")
        if not self.alphas or any(not 0 <= a <= 1 for a in self.alphas):
            raise ValueError("Select alpha values in [0, 1].")
        if not 0 <= self.categorical_tolerance < 0.5:
            raise ValueError("Categorical tolerance must be in [0, 0.5).")


@dataclass(frozen=True)
class Classification:
    """Candidate predictive models."""

    models: tuple[str, ...] = ("logistic_regression", "random_forest", "mlp")
    logistic_c: float = 1.0
    logistic_iterations: int = 300
    forest_trees: int = 100
    forest_min_leaf: int = 5
    mlp_hidden: tuple[int, ...] = (32,)
    mlp_iterations: int = 100


@dataclass(frozen=True)
class FeatureSelection:
    """Size of the feature subset to compare."""

    top_k: int = 5
    mi_neighbors: int = 3


@dataclass(frozen=True)
class Shap:
    """Sampling budgets for future SHAP analysis."""

    background_size: int = 16
    explanation_size: int = 32
    permutations: int = 16


@dataclass(frozen=True)
class Llm:
    """Opt-in explanation study; no credentials or credential URLs are stored."""

    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    repeats: int = 3
    examples: int = 2
    top_features: int = 3
    shap_run: str | None = None
    local: bool = False
    bert_model: str | None = None
    bert_layers: int = 12
    embedding_model: str | None = None
    judge_model: str | None = None

    def __post_init__(self) -> None:
        if self.repeats < 2 or min(self.examples, self.top_features, self.max_tokens) < 1:
            raise ValueError("Use at least two repeats and positive LLM study budgets.")
        if self.model and self.model == self.judge_model:
            raise ValueError("The automated judge must be distinct from the generator.")


@dataclass(frozen=True)
class Privacy:
    """Reserved evaluation settings for future privacy experiments."""

    evaluation_sample_size: int = 500


@dataclass(frozen=True)
class Visualization:
    """Default figure and table export settings."""

    figure_format: str = "pdf"
    dpi: int = 300
    decimal_places: int = 3


@dataclass(frozen=True)
class StudyConfig:
    """Immutable configuration captured with every run."""

    paths: Paths = field(default_factory=Paths)
    seed: int = 42
    variants_run: str | None = None
    models_run: str | None = None
    dataset: Dataset = field(default_factory=Dataset)
    preprocessing: Preprocessing = field(default_factory=Preprocessing)
    anonymization: Anonymization = field(default_factory=Anonymization)
    classification: Classification = field(default_factory=Classification)
    feature_selection: FeatureSelection = field(default_factory=FeatureSelection)
    shap: Shap = field(default_factory=Shap)
    llm: Llm = field(default_factory=Llm)
    privacy: Privacy = field(default_factory=Privacy)
    visualization: Visualization = field(default_factory=Visualization)

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an unsigned 32-bit integer.")
