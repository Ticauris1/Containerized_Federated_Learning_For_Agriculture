from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np # type: ignore
import torch # type: ignore


# ===================================================================
# 1. ENVIRONMENT HELPERS
# ===================================================================

def env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ===================================================================
# 2. RUNTIME
# ===================================================================

RANDOM_STATE = int(
    os.getenv(
        "RANDOM_STATE",
        "42",
    )
)

DEVICE = get_device()

ON_COLAB = "google.colab" in sys.modules
ON_MAC = sys.platform == "darwin"


# ===================================================================
# 3. PATHS
# ===================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        str(PROJECT_ROOT / "data"),
    )
).expanduser().resolve()

ORIG_ROOT = Path(
    os.getenv(
        "ORIG_ROOT",
        str(DATA_DIR / "Cotton Disease"),
    )
).expanduser().resolve()

GEOM_ROOT = Path(
    os.getenv(
        "GEOM_ROOT",
        str(DATA_DIR / "augmented_cotton_dataset_v2"),
    )
).expanduser().resolve()

GEOM_ROOT_2 = Path(
    os.getenv(
        "GEOM_ROOT_2",
        str(DATA_DIR / "combined_images"),
    )
).expanduser().resolve()


DATASETS = {
    "Original": ORIG_ROOT,
    "Geometric": GEOM_ROOT,
    "Geometric V2": GEOM_ROOT_2,
}


SAVE_DIR = Path(
    os.getenv(
        "SAVE_DIR",
        str(PROJECT_ROOT / "results"),
    )
).expanduser().resolve()

EXP_BASE_DIR = SAVE_DIR / "experiments"

SUMMARY_CSV = EXP_BASE_DIR / "_summary.csv"

MODEL_SAVE_DIR = Path(
    os.getenv(
        "MODEL_SAVE_DIR",
        str(PROJECT_ROOT / "saved_models"),
    )
).expanduser().resolve()

CACHE_DIR = Path(
    os.getenv(
        "CACHE_DIR",
        str(PROJECT_ROOT / "sk_cache"),
    )
).expanduser().resolve()


# ===================================================================
# 4. DATASET SETTINGS
# ===================================================================

RUN_ON_DATASET = os.getenv(
    "RUN_ON_DATASET",
    "Geometric V2",
)

GEOM2_MAX_IMAGES_PER_CLASS = int(
    os.getenv(
        "GEOM2_MAX_IMAGES_PER_CLASS",
        "1400",
    )
)


# ===================================================================
# 5. RUN CONTROL
# ===================================================================

RUN_MODE = os.getenv(
    "RUN_MODE",
    "deep",
).strip().lower()

RUN_FEDERATED_LEARNING = env_bool(
    "RUN_FEDERATED_LEARNING",
    True,
)

DEEP_MODELS = [
    "LW-efficientnet_b0",
]

TRADITIONAL_MODELS = [
    "SVM",
    "Naïve Bayes",
    "KNN",
    "Random Forest",
]


# ===================================================================
# 6. CENTRALIZED TRAINING
# ===================================================================

NUM_EPOCHS = 50
BATCH_SIZE = 64

BASE_LEARNING_RATE = 1e-3

FREEZE_BACKBONE = True

UNFREEZE_SCHEDULE = {
    10: 1,
    20: 2,
}


# ===================================================================
# 7. FEDERATED DISTRIBUTIONS
# ===================================================================

DIST_INDEPENDENT = (
    "Independent and Identically Distributed"
)

DIST_LABEL_SKEW = (
    "Non-Identically Distributed (Label Skew)"
)

DIST_QTY_SKEW = (
    "Non-Identically Distributed (Quantity Skew)"
)


DIST_DEFS = {
    DIST_INDEPENDENT: (
        "Each client receives samples from the same "
        "underlying distribution."
    ),

    DIST_LABEL_SKEW: (
        "Clients receive different class distributions."
    ),

    DIST_QTY_SKEW: (
        "Clients receive different quantities of samples."
    ),
}


# ===================================================================
# 8. CLIENT PARTITIONING
# ===================================================================

USE_STRATIFIED_CLIENT_SPLITS = True

LABEL_SKEW_ALPHA = 0.3

QTY_SKEW_CONCENTRATION = 0.7

MIN_CLIENT_SIZE = 100


# ===================================================================
# 9. FEDERATED TRAINING
# ===================================================================

FED_ROUNDS = 50

FED_LOCAL_EPOCHS = 2

FED_LEARNING_RATE = 1e-3

FED_COLD_START = env_bool(
    "FED_COLD_START",
    True,
)


TRAIN_AGG_MODE = "weight_avg"

EVAL_AGG_MODE = "weight_avg"

PREDICTION_WEIGHT_MODE = "data_size"


# FedAdam

FEDADAM_SERVER_LR = 1e-3

FEDADAM_BETA1 = 0.9

FEDADAM_BETA2 = 0.999

FEDADAM_EPS = 1e-8


# FedProx

FEDPROX_MU = 0.01


# ===================================================================
# 10. FEDERATED EXPERIMENT GRID
# ===================================================================

EXPERIMENT_CLIENT_COUNTS = [
    2,
]

EXPERIMENT_DISTRIBUTIONS = [
    DIST_LABEL_SKEW,
]

EXPERIMENT_PARTICIPATION_RATES = [
    1.0,
]

EXPERIMENT_METHODS = [
    "FedAdam",
    "FedProx",
]


# Temporary compatibility for distributed runtime.
FED_CLIENTS_PER_ROUND = 2


# ===================================================================
# 11. TRADITIONAL ML
# ===================================================================

SCORING = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "nll": "neg_log_loss",
}

REFIT_METRIC = "f1"

USE_FLATTENED_PIXELS_FOR_TRAD = True

PIXEL_IMG_SIZE = 32

SVM_BACKEND = "kpca"


# ===================================================================
# 12. HYBRID INFERENCE
# ===================================================================

USE_HYBRID_INFERENCE = env_bool(
    "USE_HYBRID_INFERENCE",
    False,
)

USE_BINARY_HYBRID_EVAL = True

HYBRID_POSITIVE_CLASSES = [
    0,
    1,
]

HYBRID_THRESHOLD_HIGH = 0.60

HYBRID_EVAL_SPLIT = "test"

HYBRID_SAVE_DETAILS = True


# ===================================================================
# 13. BAYESIAN OPTIMIZATION
# ===================================================================

USE_HYBRID_BAYES_OPT = env_bool(
    "USE_HYBRID_BAYES_OPT",
    False,
)

HYBRID_BO_THRESHOLD_MIN = 0.75

HYBRID_BO_THRESHOLD_MAX = 0.995

HYBRID_BO_N_INIT = 5

HYBRID_BO_N_ITER = 20

HYBRID_BO_MAX_ACC_DROP = 0.01

HYBRID_BO_FALLBACK_THRESHOLD = 0.80

HYBRID_PI_K = 10.0

HYBRID_BO_ACCURACY_PENALTY = 50.0


# ===================================================================
# 14. LIGHTWEIGHT MODEL
# ===================================================================

LIGHTWEIGHT_MODEL_NAME = "paper_lightweight"

LIGHTWEIGHT_MODEL_CHECKPOINT = Path(
    os.getenv(
        "LIGHTWEIGHT_MODEL_CHECKPOINT",
        str(
            PROJECT_ROOT
            / "models"
            / "paper_lightweight"
            / "paper_lightweight_best.pth"
        ),
    )
).expanduser().resolve()


# ===================================================================
# 15. PRIVACY-AWARE AGGREGATION
# ===================================================================

USE_PRIVACY_AWARE_AGGREGATION = env_bool(
    "USE_PRIVACY_AWARE_AGGREGATION",
    True,
)

PRIVACY_AGG_TARGET = "classifier_bias"

PRIVACY_AGG_SCALE = 10_000

PRIVACY_AGG_MODULUS = 1_000_003

PRIVACY_AGG_SEED = RANDOM_STATE


# ===================================================================
# 16. INITIALIZATION
# ===================================================================

def initialize_runtime() -> None:
    """Initialize deterministic runtime state and directories."""

    set_seed(RANDOM_STATE)

    directories = (
        SAVE_DIR,
        EXP_BASE_DIR,
        MODEL_SAVE_DIR,
        CACHE_DIR,
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ===================================================================
# 17. VALIDATION
# ===================================================================

def validate_configuration() -> None:
    """Validate configuration before running experiments."""

    valid_modes = {
        "deep",
        "traditional",
        "both",
    }

    if RUN_MODE not in valid_modes:
        raise ValueError(
            f"RUN_MODE must be one of "
            f"{sorted(valid_modes)}. "
            f"Received: {RUN_MODE!r}"
        )

    if (
        RUN_ON_DATASET != "Both"
        and RUN_ON_DATASET not in DATASETS
    ):
        raise ValueError(
            f"Unknown dataset: {RUN_ON_DATASET!r}. "
            f"Available: {list(DATASETS)}"
        )

    if BATCH_SIZE <= 0:
        raise ValueError(
            "BATCH_SIZE must be greater than zero."
        )

    if FED_ROUNDS <= 0:
        raise ValueError(
            "FED_ROUNDS must be greater than zero."
        )

    if FED_LOCAL_EPOCHS <= 0:
        raise ValueError(
            "FED_LOCAL_EPOCHS must be greater than zero."
        )

    if LABEL_SKEW_ALPHA <= 0:
        raise ValueError(
            "LABEL_SKEW_ALPHA must be greater than zero."
        )

    if QTY_SKEW_CONCENTRATION <= 0:
        raise ValueError(
            "QTY_SKEW_CONCENTRATION must be greater than zero."
        )

    if MIN_CLIENT_SIZE <= 0:
        raise ValueError(
            "MIN_CLIENT_SIZE must be greater than zero."
        )

    if not (
        0
        <= HYBRID_BO_THRESHOLD_MIN
        < HYBRID_BO_THRESHOLD_MAX
        <= 1
    ):
        raise ValueError(
            "Hybrid threshold range must satisfy "
            "0 <= min < max <= 1."
        )

    if not (
        0
        <= HYBRID_BO_MAX_ACC_DROP
        <= 1
    ):
        raise ValueError(
            "HYBRID_BO_MAX_ACC_DROP must be "
            "between 0 and 1."
        )

    for rate in EXPERIMENT_PARTICIPATION_RATES:
        if not 0 < rate <= 1:
            raise ValueError(
                "Experiment participation rates "
                "must be in (0, 1]."
            )

    for clients in EXPERIMENT_CLIENT_COUNTS:
        if clients <= 0:
            raise ValueError(
                "Experiment client counts "
                "must be greater than zero."
            )


# ===================================================================
# 18. SUMMARY
# ===================================================================

def print_configuration_summary() -> None:
    """Print active configuration."""

    if ON_MAC:
        platform = "Mac"

    elif ON_COLAB:
        platform = "Colab"

    else:
        platform = sys.platform

    print("\n" + "=" * 60)
    print("🚀 CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"Device:                {DEVICE.type.upper()}")
    print(f"Platform:              {platform}")
    print(f"Project root:          {PROJECT_ROOT}")
    print(f"Random state:          {RANDOM_STATE}")

    print("\nDATA")
    print(f"Dataset:               {RUN_ON_DATASET}")
    print(f"Mode:                  {RUN_MODE}")

    print("\nFEDERATED TRAINING")
    print(f"Enabled:               {RUN_FEDERATED_LEARNING}")
    print(f"Rounds:                {FED_ROUNDS}")
    print(f"Local epochs:          {FED_LOCAL_EPOCHS}")
    print(f"Learning rate:         {FED_LEARNING_RATE}")

    print("\nEXPERIMENT GRID")
    print(f"Clients:               {EXPERIMENT_CLIENT_COUNTS}")
    print(f"Distributions:         {EXPERIMENT_DISTRIBUTIONS}")
    print(f"Participation:         {EXPERIMENT_PARTICIPATION_RATES}")
    print(f"Methods:               {EXPERIMENT_METHODS}")

    print("\nMODELS")
    print(f"Deep:                  {DEEP_MODELS}")
    print(f"Traditional:           {TRADITIONAL_MODELS}")

    print("\nHYBRID")
    print(f"Enabled:               {USE_HYBRID_INFERENCE}")
    print(f"Bayesian optimization: {USE_HYBRID_BAYES_OPT}")

    print("\nPRIVACY")
    print(
        f"Aggregation enabled:   "
        f"{USE_PRIVACY_AWARE_AGGREGATION}"
    )

    print("=" * 60 + "\n")