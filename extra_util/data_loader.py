from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetConfig:
    name: str = "wine_quality"
    target: str = ""
    csv_path: str = ""
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    scale_numeric: bool = True
    quality_threshold: int = 7
    wine_variant: str = "both" 


def load_dataset(cfg: DatasetConfig):
    if cfg.name == "wine_quality":
        # Load UCI Wine Quality datasets (CSV via HTTP). Target: quality ∈ {3..8}
        urls = {
            "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
        }
        frames = []
        if cfg.wine_variant in ("red", "both"):
            df_red = pd.read_csv(urls["red"], sep=';')
            df_red["variant"] = 0  # optional indicator
            frames.append(df_red)
        if cfg.wine_variant in ("white", "both"):
            df_white = pd.read_csv(urls["white"], sep=';')
            df_white["variant"] = 1
            frames.append(df_white)
        if not frames:
            raise ValueError("wine_variant must be one of {'red','white','both'}")
        df = pd.concat(frames, ignore_index=True)

        # Binarize target by threshold on quality
        if "quality" not in df.columns:
            raise ValueError("Wine dataset missing 'quality' column")
        y = (df["quality"] >= cfg.quality_threshold).astype(int)
        y = pd.Series(y.values, name="target")
        X = df.drop(columns=["quality"])  
    else:
        raise ValueError(f"Unknown dataset name: {cfg.name}")
    return X, y


def preprocess_and_split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        val_size: float,
        random_state: int,
        scale_numeric: bool = True
):
    # First split off (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    # Then split temp into val and test with correct proportions
    test_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, stratify=y_temp, random_state=random_state
    )

    if scale_numeric:
        scaler = StandardScaler()
        num_cols = X_train.select_dtypes(include=["number"]).columns
        # ensure float dtype to avoid incompatible assignment warnings
        X_train[num_cols] = X_train[num_cols].astype("float64")
        X_val[num_cols] = X_val[num_cols].astype("float64")
        X_test[num_cols] = X_test[num_cols].astype("float64")
        # fit on train; transform all and assign via DataFrame to preserve dtypes/index
        X_train[num_cols] = pd.DataFrame(
            scaler.fit_transform(X_train[num_cols].to_numpy()),
            columns=num_cols, index=X_train.index
        )
        X_val[num_cols] = pd.DataFrame(
            scaler.transform(X_val[num_cols].to_numpy()),
            columns=num_cols, index=X_val.index
        )
        X_test[num_cols] = pd.DataFrame(
            scaler.transform(X_test[num_cols].to_numpy()),
            columns=num_cols, index=X_test.index
        )

    return X_train, X_val, X_test, y_train, y_val, y_test
