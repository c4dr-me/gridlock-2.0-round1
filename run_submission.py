# ==========================================================================================
# SUBMISSION-READY STANDALONE PIPELINE
# Raw AR cluster decoder + hyperlocal day48/day49 correction. Writes one final submission.
# ==========================================================================================

import os
import json
import zipfile
import random
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

VERSION = "submission_ready_9231_v267"
FINAL_SUBMISSION_NAME = "submission_Jayant.csv"


class MLELog:
    """MLE Dashboard logging helper with premium visual styling."""
    
    @staticmethod
    def header(title):
        """Prints a large ASCII section header banner.
        
        Args:
            title (str): The text to display inside the banner.
        """
        width = 80
        border = "=" * width
        print(f"\n{border}")
        print(f"| {title.center(width - 4)} |")
        print(f"{border}\n")

    @staticmethod
    def step(step_num, total_steps, description):
        """Prints a pipeline process indicator.
        
        Args:
            step_num (int): Current step index.
            total_steps (int): Total steps in pipeline.
            description (str): Description of the step.
        """
        print(f"[{step_num}/{total_steps}] --- {description} ---")

    @staticmethod
    def info(msg):
        """Prints a structured information message.
        
        Args:
            msg (str): Information message.
        """
        print(f"   [INFO] {msg}")

    @staticmethod
    def success(msg):
        """Prints a success message.
        
        Args:
            msg (str): Success message.
        """
        print(f"   [SUCCESS] {msg}")

    @staticmethod
    def warning(msg):
        """Prints a warning message.
        
        Args:
            msg (str): Warning message.
        """
        print(f"   [WARNING] {msg}")

    @staticmethod
    def print_table(title, data_dict):
        """Prints a beautiful table from a flat dictionary of metrics.
        
        Args:
            title (str): The table header title.
            data_dict (dict): Flat dictionary containing metrics.
        """
        width = 60
        border = "-" * width
        print(f"\n+{border}+")
        print(f"| {title.center(width - 2)} |")
        print(f"+{border}+")
        for k, v in data_dict.items():
            if isinstance(v, float):
                v_str = f"{v:.6f}"
            else:
                v_str = str(v)
            print(f"| {k:<30} | {v_str:>23} |")
        print(f"+{border}+\n")

    @staticmethod
    def print_dataframe(title, df):
        """Prints a pandas DataFrame formatted as a beautiful ASCII table.
        
        Args:
            title (str): The table header title.
            df (pd.DataFrame): The pandas DataFrame to format.
        """
        cols = list(df.columns)
        index_col = df.index.name if df.index.name else "index"
        
        rows = []
        for idx, row in df.iterrows():
            rows.append([str(idx)] + [f"{val:.5f}" if isinstance(val, (float, np.floating)) else str(val) for val in row])
        
        headers = [index_col] + cols
        col_widths = []
        for i, h in enumerate(headers):
            max_r = max(len(r[i]) for r in rows) if rows else 0
            col_widths.append(max(len(h), max_r))
            
        border_parts = ["-" * (w + 2) for w in col_widths]
        border = "+" + "+".join(border_parts) + "+"
        
        total_width = len(border)
        print(f"\n+{'=' * (total_width - 2)}+")
        print(f"| {title.center(total_width - 4)} |")
        print(f"+{'=' * (total_width - 2)}+")
        print(border)
        
        header_line = "|"
        for i, h in enumerate(headers):
            header_line += f" {h.center(col_widths[i])} |"
        print(header_line)
        print(border)
        
        for r in rows:
            row_line = "|"
            for i, val in enumerate(r):
                if i > 0:
                    row_line += f" {val.rjust(col_widths[i])} |"
                else:
                    row_line += f" {val.ljust(col_widths[i])} |"
            print(row_line)
        print(border + "\n")


# ==========================================================================================
# --- STAGE 1: Initialization & Helpers ---
# ==========================================================================================


def find_dataset_dir():
    """Locates the directory containing the dataset files (train.csv and test.csv).
    
    Searches through a predefined list of candidate paths on the filesystem to
    find the directory where the train and test CSV files reside.
    
    Returns:
        Path: The Path object of the directory containing the dataset files.
        
    Raises:
        FileNotFoundError: If train.csv and test.csv are not found in any candidate path.
    """
    candidates = [
        Path(r"E:\ROCm\Flipkart\Submission"),
        Path(r"E:\ROCm\Flipkart"),
        Path.cwd(),
        Path("/kaggle/input/datasets/c4drme/flipkart"),
        Path("/kaggle/input/datasets/c4drme"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    for root in candidates:
        if root.exists() and (root / "train.csv").exists() and (root / "test.csv").exists():
            return root
    for root in candidates:
        if root.exists():
            train_hits = list(root.rglob("train.csv"))
            for train_path in train_hits:
                test_path = train_path.parent / "test.csv"
                if test_path.exists():
                    return train_path.parent
    raise FileNotFoundError("Could not find train.csv and test.csv. Put them beside this notebook/script or in the Kaggle input folder.")


DATA_DIR = find_dataset_dir()
TRAIN_PATH = str(DATA_DIR / "train.csv")
TEST_PATH = str(DATA_DIR / "test.csv")
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

if Path("/kaggle/working").exists():
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = str(DATA_DIR)

os.makedirs(OUT_DIR, exist_ok=True)


def seed_everything(seed=42):
    """Sets random seeds for reproducibility across various libraries.
    
    Args:
        seed (int): The seed value to set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)

MLELog.header("SUBMISSION-READY 92.31 V267 PIPELINE")
MLELog.step(1, 5, "Initializing & Loading Data")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
MLELog.info(f"Loaded Train shape: {train_df.shape} | Test shape: {test_df.shape}")


def decode_geohash(geohash):
    """Decodes a standard geohash string into latitude and longitude coordinates.
    
    Args:
        geohash (str): The geohash string to decode.
        
    Returns:
        tuple[float, float]: A tuple containing the decoded (latitude, longitude).
    """
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    base32_map = {char: i for i, char in enumerate(BASE32)}
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True

    for char in str(geohash):
        if char not in base32_map:
            continue
        val = base32_map[char]
        for mask in [16, 8, 4, 2, 1]:
            bit = 1 if (val & mask) else 0
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if bit:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if bit:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even

    return (lat_interval[0] + lat_interval[1]) / 2.0, (lon_interval[0] + lon_interval[1]) / 2.0


def encode_fourier_harmonics(df, minutes_col, cycles=[1440, 720, 480, 360]):
    """Encodes a time column (in minutes) into cyclical sine and cosine Fourier harmonics.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        minutes_col (str): The column name containing the time in minutes.
        cycles (list[int]): List of cycle lengths (in minutes) for harmonic mapping.
            Defaults to [1440, 720, 480, 360].
            
    Returns:
        pd.DataFrame: A copy of the DataFrame with added sine and cosine harmonic columns.
    """
    df = df.copy()
    for cycle in cycles:
        angle = 2.0 * np.pi * df[minutes_col] / cycle
        df[f"sin_{cycle}"] = np.sin(angle)
        df[f"cos_{cycle}"] = np.cos(angle)
    return df


def add_oof_target_encodings(train_df, test_df, col, target_col, n_splits=5, smoothing_val=10.0):
    """Computes Out-Of-Fold (OOF) target encodings for a categorical column.
    
    Uses K-Fold splits on the training data to calculate smooth target encodings to
    prevent leakage, and maps the full training statistics onto the test data.
    
    Args:
        train_df (pd.DataFrame): Training features DataFrame.
        test_df (pd.DataFrame): Test features DataFrame.
        col (str): The column name to encode.
        target_col (str): The target column name.
        n_splits (int): Number of folds for OOF splitting. Defaults to 5.
        smoothing_val (float): Smoothing factor value. Defaults to 10.0.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The updated (train_df, test_df) with encoding columns.
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    train_copy[f"{col}_te"] = np.nan
    test_copy[f"{col}_te"] = np.nan

    global_mean = train_copy[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(train_copy):
        fold_train = train_copy.iloc[tr_idx]
        stats = fold_train.groupby(col)[target_col].agg(["count", "mean"])
        encoded = (stats["count"] * stats["mean"] + smoothing_val * global_mean) / (stats["count"] + smoothing_val)
        train_copy.iloc[va_idx, train_copy.columns.get_loc(f"{col}_te")] = (
            train_copy.iloc[va_idx][col].map(encoded.to_dict()).fillna(global_mean)
        )

    stats_full = train_copy.groupby(col)[target_col].agg(["count", "mean"])
    encoded_full = (stats_full["count"] * stats_full["mean"] + smoothing_val * global_mean) / (stats_full["count"] + smoothing_val)
    test_copy[f"{col}_te"] = test_copy[col].map(encoded_full.to_dict()).fillna(global_mean)
    train_copy[f"{col}_te"] = train_copy[f"{col}_te"].fillna(global_mean)

    return train_copy, test_copy


def add_oof_historical_geohash_stats(train_df, test_df, n_splits=5):
    """Computes Out-Of-Fold (OOF) historical demand statistics for geohashes.
    
    Calculates historical metrics including mean demand, std demand, max demand,
    and spike rate for geohashes using K-Fold split mapping.
    
    Args:
        train_df (pd.DataFrame): Training features DataFrame.
        test_df (pd.DataFrame): Test features DataFrame.
        n_splits (int): Number of folds for OOF splitting. Defaults to 5.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The updated (train_df, test_df) with statistic columns.
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    for col in ["geohash_mean_demand", "geohash_std_demand", "geohash_max_demand", "geohash_spike_rate"]:
        train_copy[col] = np.nan
        test_copy[col] = np.nan

    train_copy["is_spike"] = (train_copy["demand"] == 1.0).astype(int)

    global_mean = train_copy["demand"].mean()
    global_std = train_copy["demand"].std()
    global_max = train_copy["demand"].max()
    global_spike_rate = train_copy["is_spike"].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, va_idx in kf.split(train_copy):
        fold_train = train_copy.iloc[tr_idx]
        stats = fold_train.groupby("geohash")["demand"].agg(["mean", "std", "max"]).reset_index()
        stats.columns = ["geohash", "mean_val", "std_val", "max_val"]
        spike = fold_train.groupby("geohash")["is_spike"].mean().reset_index()
        spike.columns = ["geohash", "spike_rate_val"]
        stats = stats.merge(spike, on="geohash")
        stats["std_val"] = stats["std_val"].fillna(0.0)

        val_fold = train_copy.iloc[va_idx][["geohash"]].merge(stats, on="geohash", how="left")
        val_fold["mean_val"] = val_fold["mean_val"].fillna(global_mean)
        val_fold["std_val"] = val_fold["std_val"].fillna(global_std)
        val_fold["max_val"] = val_fold["max_val"].fillna(global_max)
        val_fold["spike_rate_val"] = val_fold["spike_rate_val"].fillna(global_spike_rate)

        train_copy.iloc[va_idx, train_copy.columns.get_loc("geohash_mean_demand")] = val_fold["mean_val"].values
        train_copy.iloc[va_idx, train_copy.columns.get_loc("geohash_std_demand")] = val_fold["std_val"].values
        train_copy.iloc[va_idx, train_copy.columns.get_loc("geohash_max_demand")] = val_fold["max_val"].values
        train_copy.iloc[va_idx, train_copy.columns.get_loc("geohash_spike_rate")] = val_fold["spike_rate_val"].values

    stats_full = train_copy.groupby("geohash")["demand"].agg(["mean", "std", "max"]).reset_index()
    stats_full.columns = ["geohash", "mean_val", "std_val", "max_val"]
    spike_full = train_copy.groupby("geohash")["is_spike"].mean().reset_index()
    spike_full.columns = ["geohash", "spike_rate_val"]
    stats_full = stats_full.merge(spike_full, on="geohash")
    stats_full["std_val"] = stats_full["std_val"].fillna(0.0)

    test_fold = test_copy[["geohash"]].merge(stats_full, on="geohash", how="left")
    test_copy["geohash_mean_demand"] = test_fold["mean_val"].fillna(global_mean).values
    test_copy["geohash_std_demand"] = test_fold["std_val"].fillna(global_std).values
    test_copy["geohash_max_demand"] = test_fold["max_val"].fillna(global_max).values
    test_copy["geohash_spike_rate"] = test_fold["spike_rate_val"].fillna(global_spike_rate).values

    train_copy["geohash_time_block_spike_rate"] = np.nan
    test_copy["geohash_time_block_spike_rate"] = np.nan

    for tr_idx, va_idx in kf.split(train_copy):
        fold_train = train_copy.iloc[tr_idx]
        tb_spike = fold_train.groupby("geohash_time_block")["is_spike"].mean().to_dict()
        train_copy.iloc[va_idx, train_copy.columns.get_loc("geohash_time_block_spike_rate")] = (
            train_copy.iloc[va_idx]["geohash_time_block"].map(tb_spike).fillna(global_spike_rate)
        )

    tb_spike_full = train_copy.groupby("geohash_time_block")["is_spike"].mean().to_dict()
    test_copy["geohash_time_block_spike_rate"] = test_copy["geohash_time_block"].map(tb_spike_full).fillna(global_spike_rate)

    train_copy = train_copy.drop(columns=["is_spike"])
    return train_copy, test_copy


def compute_multiscale_spatial_centrality(train_df, test_df, k_values=[3, 10, 20]):
    """Calculates spatial centrality scores using K-Nearest Neighbors at multiple scales.
    
    Decodes geohashes to get coordinates and defines spatial density indicators
    by calculating inverse average distances to nearest neighbors.
    
    Args:
        train_df (pd.DataFrame): Training features DataFrame.
        test_df (pd.DataFrame): Test features DataFrame.
        k_values (list[int]): List of neighbor counts (k) to compute centrality.
            Defaults to [3, 10, 20].
            
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The updated (train_df, test_df) with centrality features.
    """
    train_copy = train_df.copy()
    test_copy = test_df.copy()

    unique_ghs = pd.concat([train_copy["geohash"], test_copy["geohash"]]).unique()
    coord_map = {gh: decode_geohash(gh) for gh in unique_ghs}

    for df in [train_copy, test_copy]:
        df["latitude"] = df["geohash"].map(lambda x: coord_map[x][0])
        df["longitude"] = df["geohash"].map(lambda x: coord_map[x][1])

    unique = train_copy[["geohash", "latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    coords = unique[["latitude", "longitude"]].values

    for k in k_values:
        n_neighbors = min(k + 1, len(coords))
        if n_neighbors <= 1:
            train_copy[f"spatial_centrality_{k}"] = 1.0
            test_copy[f"spatial_centrality_{k}"] = 1.0
            continue

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(coords)
        distances, _ = nn.kneighbors(coords)
        avg_dist = distances[:, 1:].mean(axis=1)

        unique[f"spatial_centrality_{k}"] = 1.0 / (avg_dist + 1e-5)
        centrality_map = unique.set_index("geohash")[f"spatial_centrality_{k}"].to_dict()
        global_avg = unique[f"spatial_centrality_{k}"].mean()

        train_copy[f"spatial_centrality_{k}"] = train_copy["geohash"].map(centrality_map).fillna(global_avg)
        test_copy[f"spatial_centrality_{k}"] = test_copy["geohash"].map(centrality_map).fillna(global_avg)

    return train_copy, test_copy


# ==========================================================================================
# --- STAGE 2: Feature Engineering Pipeline ---
# ==========================================================================================


def pipeline_feature_engineering(train, test):
    """Executes the full feature engineering pipeline for model training and prediction.
    
    Processes the raw datasets, parses timestamps, creates cyclic features, spillovers,
    spatial centralities, lags, target encodings, and handles missing values.
    
    Args:
        train (pd.DataFrame): Raw train DataFrame.
        test (pd.DataFrame): Raw test DataFrame.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The preprocessed (train, test) DataFrames.
    """
    MLELog.step(2, 5, "Executing Feature Engineering Pipeline")
    train_clean = train.copy()
    test_clean = test.copy()

    train_clean["idle_capacity"] = 1.0 - train_clean["demand"]

    for df in [train_clean, test_clean]:
        df["interval_key"] = df["day"].astype(str) + "_" + df["timestamp"].astype(str)
        df["minutes"] = df["timestamp"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
        df["day_of_week"] = (df["day"] - 1) % 7
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["time_block"] = df["minutes"] // 180
        df["time_block_1h"] = df["minutes"] // 60
        df["geohash_time_block"] = df["geohash"].astype(str) + "_" + df["time_block"].astype(str)
        df["geohash_time_block_1h"] = df["geohash"].astype(str) + "_" + df["time_block_1h"].astype(str)
        df["geohash_weather"] = df["geohash"].astype(str) + "_" + df["Weather"].astype(str)

    train_clean = encode_fourier_harmonics(train_clean, "minutes")
    test_clean = encode_fourier_harmonics(test_clean, "minutes")

    MLELog.info("Computing cross-sectional features...")
    combined_temp = pd.concat([train_clean, test_clean], ignore_index=True)
    combined_temp["interval_active_logs"] = combined_temp.groupby("interval_key")["geohash"].transform("count")
    combined_temp["interval_avg_temp"] = combined_temp.groupby("interval_key")["Temperature"].transform("mean")
    combined_temp["interval_avg_temp"] = combined_temp["interval_avg_temp"].fillna(combined_temp["Temperature"].median())

    MLELog.info("Computing neighborhood activity spillover lags...")
    unique_ghs = combined_temp["geohash"].unique()
    coord_map = {gh: decode_geohash(gh) for gh in unique_ghs}

    unique = combined_temp[["geohash"]].drop_duplicates().reset_index(drop=True)
    unique["latitude"] = unique["geohash"].map(lambda x: coord_map[x][0])
    unique["longitude"] = unique["geohash"].map(lambda x: coord_map[x][1])
    coords = unique[["latitude", "longitude"]].values

    nn = NearestNeighbors(n_neighbors=min(4, len(coords)), metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    gh_list = unique["geohash"].values
    neighbor_map = {}
    for i, gh in enumerate(gh_list):
        neighbor_map[gh] = [gh_list[idx] for idx in indices[i, 1:] if idx < len(gh_list)]

    for idx in range(3):
        combined_temp[f"neighbor_{idx}"] = combined_temp["geohash"].map(
            lambda x: neighbor_map.get(x, [x, x, x])[idx] if len(neighbor_map.get(x, [])) > idx else x
        )

    active_pairs = combined_temp[["interval_key", "geohash"]].drop_duplicates().copy()
    active_pairs["is_active"] = 1

    for idx in range(3):
        active_pairs_idx = active_pairs.rename(
            columns={"is_active": f"is_active_neigh_{idx}", "geohash": f"geohash_neigh_{idx}"}
        )
        combined_temp = combined_temp.merge(
            active_pairs_idx,
            left_on=["interval_key", f"neighbor_{idx}"],
            right_on=["interval_key", f"geohash_neigh_{idx}"],
            how="left",
        )
        combined_temp[f"is_active_neigh_{idx}"] = combined_temp[f"is_active_neigh_{idx}"].fillna(0)
        if f"geohash_neigh_{idx}" in combined_temp.columns:
            combined_temp = combined_temp.drop(columns=[f"geohash_neigh_{idx}"])

    combined_temp["neighbor_active_ratio"] = (
        combined_temp["is_active_neigh_0"] + combined_temp["is_active_neigh_1"] + combined_temp["is_active_neigh_2"]
    ) / 3.0

    combined_temp = combined_temp.drop(
        columns=[
            "interval_key",
            "neighbor_0",
            "neighbor_1",
            "neighbor_2",
            "is_active_neigh_0",
            "is_active_neigh_1",
            "is_active_neigh_2",
        ]
    )

    train_clean = combined_temp.iloc[:len(train)].copy().reset_index(drop=True)
    test_clean = combined_temp.iloc[len(train):].copy().reset_index(drop=True)

    MLELog.info("Computing spatial centralities...")
    train_clean, test_clean = compute_multiscale_spatial_centrality(train_clean, test_clean)

    MLELog.info("Computing chronological features...")
    for df in [train_clean, test_clean]:
        df["abs_minutes"] = (df["day"] - 1) * 1440 + df["minutes"]

    train_clean["is_test"] = 0
    test_clean["is_test"] = 1
    train_clean["orig_index"] = train_clean.index
    test_clean["orig_index"] = test_clean.index

    combined_time = pd.concat([train_clean, test_clean], ignore_index=True)
    combined_time = combined_time.sort_values(["geohash", "abs_minutes"]).reset_index(drop=True)

    combined_time["time_delta_last_log"] = combined_time.groupby("geohash")["abs_minutes"].diff().fillna(1440.0)
    combined_time["demand_lag1"] = combined_time.groupby("geohash")["demand"].shift(1).fillna(0.0)
    combined_time["ewma_03"] = (
        combined_time.groupby("geohash")["demand"]
        .apply(lambda x: x.shift(1).ewm(alpha=0.3, min_periods=1).mean())
        .fillna(0.0)
        .reset_index(drop=True)
    )

    train_clean = combined_time[combined_time["is_test"] == 0].sort_values("orig_index").reset_index(drop=True)
    test_clean = combined_time[combined_time["is_test"] == 1].sort_values("orig_index").reset_index(drop=True)
    train_clean = train_clean.drop(columns=["is_test", "orig_index"])
    test_clean = test_clean.drop(columns=["is_test", "orig_index"])

    MLELog.info("Computing OOF target encodings...")
    for col in ["Weather", "RoadType", "geohash_time_block", "geohash_time_block_1h", "geohash_weather"]:
        train_clean, test_clean = add_oof_target_encodings(train_clean, test_clean, col, "idle_capacity")

    MLELog.info("Computing OOF historical geohash stats...")
    train_clean, test_clean = add_oof_historical_geohash_stats(train_clean, test_clean)

    for df in [train_clean, test_clean]:
        df["LargeVehicles_bin"] = (df["LargeVehicles"] == "Allowed").astype(int)
        df["Landmarks_bin"] = (df["Landmarks"] == "Yes").astype(int)
        df["Bottleneck_Index"] = df["LargeVehicles_bin"] / df["NumberofLanes"].clip(lower=1)
        for col in ["Weather", "RoadType"]:
            df[col] = df[col].fillna("Missing")

    for col in ["NumberofLanes", "Temperature"]:
        med = train_clean[col].median()
        train_clean[col] = train_clean[col].fillna(med)
        test_clean[col] = test_clean[col].fillna(med)

    MLELog.success("Feature engineering completed successfully.")
    return train_clean, test_clean


train_processed, test_processed = pipeline_feature_engineering(train_df, test_df)


# ==========================================================================================
# --- STAGE 3: Geohash Clustering ---
# ==========================================================================================

MLELog.step(3, 5, "Building Geohash Clusters")

geo_stats = train_processed.groupby("geohash").agg(
    mean_demand=("demand", "mean"),
    std_demand=("demand", "std"),
    max_demand=("demand", "max"),
    spike_rate=("demand", lambda x: float((x >= 0.95).mean())),
    count=("demand", "count"),
    lanes=("NumberofLanes", "mean"),
    temp=("Temperature", "mean"),
    centrality=("spatial_centrality_10", "mean"),
    active=("neighbor_active_ratio", "mean"),
).reset_index()

geo_stats["std_demand"] = geo_stats["std_demand"].fillna(0.0)
cluster_features = [
    "mean_demand",
    "std_demand",
    "max_demand",
    "spike_rate",
    "count",
    "lanes",
    "temp",
    "centrality",
    "active",
]

Xc = geo_stats[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
Xc = StandardScaler().fit_transform(Xc)

km = KMeans(n_clusters=3, random_state=42, n_init=20)
raw_cluster = km.fit_predict(Xc)
geo_stats["raw_cluster"] = raw_cluster

cluster_summary = geo_stats.groupby("raw_cluster").agg(
    mean_demand=("mean_demand", "mean"),
    max_demand=("max_demand", "mean"),
    spike_rate=("spike_rate", "mean"),
    count=("count", "mean"),
    std_demand=("std_demand", "mean"),
).reset_index()

cluster_summary["intensity"] = (
    1.20 * cluster_summary["mean_demand"]
    + 0.70 * cluster_summary["max_demand"]
    + 0.90 * cluster_summary["spike_rate"]
    + 0.10 * np.log1p(cluster_summary["count"])
)

ordered = cluster_summary.sort_values("intensity")["raw_cluster"].tolist()
remap = {
    ordered[0]: 0,
    ordered[-1]: 1,
    ordered[1]: 2,
}

geo_stats["cluster"] = geo_stats["raw_cluster"].map(remap).astype(int)
cluster_map = geo_stats.set_index("geohash")["cluster"].to_dict()

MLELog.print_dataframe("Geohash Cluster Summary (Profiles)", geo_stats.groupby("cluster")[cluster_features].mean())


# ==========================================================================================
# --- STAGE 4: Model Training ---
# ==========================================================================================

MLELog.step(4, 5, "Training Predictive Models Ensemble")


def asym_mse_xgb(preds, dtrain):
    """Custom asymmetric Mean Squared Error objective function for XGBoost.
    
    Penalizes underpredictions heavily when the label is above a threshold.
    
    Args:
        preds (np.ndarray): Predicted values.
        dtrain (xgb.DMatrix): Training data matrix containing labels.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Gradient and Hessian values.
    """
    labels = dtrain.get_label()
    grad = preds - labels
    mask = (labels > 0.7) & (preds < labels)
    grad[mask] *= 20.0
    hess = np.ones_like(labels)
    hess[mask] *= 20.0
    return grad, hess


def asym_mse_lgb(labels, preds):
    """Custom asymmetric Mean Squared Error objective function for LightGBM.
    
    Penalizes underpredictions heavily when the label is above a threshold.
    
    Args:
        labels (np.ndarray): Ground truth labels.
        preds (np.ndarray): Predicted values.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Gradient and Hessian values.
    """
    grad = preds - labels
    mask = (labels > 0.7) & (preds < labels)
    grad[mask] *= 20.0
    hess = np.ones_like(labels)
    hess[mask] *= 20.0
    return grad, hess


cat_features = ["geohash", "day_of_week", "RoadType", "Weather", "LargeVehicles", "Landmarks"]
for col in cat_features:
    train_processed[col] = train_processed[col].astype(str)
    test_processed[col] = test_processed[col].astype(str)

numeric_features = [
    "day", "is_weekend", "time_block",
    "latitude", "longitude",
    "spatial_centrality_3", "spatial_centrality_10", "spatial_centrality_20",
    "time_delta_last_log", "demand_lag1", "ewma_03",
    "minutes", "sin_1440", "cos_1440", "sin_720", "cos_720",
    "sin_480", "cos_480", "sin_360", "cos_360",
    "NumberofLanes", "LargeVehicles_bin", "Landmarks_bin",
    "Bottleneck_Index", "Temperature",
    "interval_active_logs", "interval_avg_temp", "neighbor_active_ratio",
    "Weather_te", "RoadType_te", "geohash_time_block_te",
    "geohash_mean_demand", "geohash_std_demand", "geohash_max_demand",
    "geohash_spike_rate", "geohash_time_block_spike_rate",
]

features = cat_features + numeric_features

train_processed_xgb = train_processed.copy()
test_processed_xgb = test_processed.copy()
for col in cat_features:
    train_processed_xgb[col] = train_processed_xgb[col].astype("category")
    test_processed_xgb[col] = test_processed_xgb[col].astype("category")

MLELog.info("Training CatBoost seed 42...")
cb_model1 = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.9",
    iterations=2000,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=5.0,
    random_seed=42,
    verbose=False,
)
cb_model1.fit(train_processed[features], train_processed["idle_capacity"], cat_features=cat_features)

MLELog.info("Training CatBoost seed 777...")
cb_model2 = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.9",
    iterations=2000,
    learning_rate=0.025,
    depth=7,
    l2_leaf_reg=5.0,
    random_seed=777,
    verbose=False,
)
cb_model2.fit(train_processed[features], train_processed["idle_capacity"], cat_features=cat_features)

MLELog.info("Training LightGBM...")
lgb_model = LGBMRegressor(
    n_estimators=2400,
    learning_rate=0.012,
    num_leaves=63,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    objective=asym_mse_lgb,
)
lgb_model.fit(train_processed_xgb[features], train_processed_xgb["demand"])

MLELog.info("Training XGBoost...")
dtrain = xgb.DMatrix(train_processed_xgb[features], label=train_processed_xgb["demand"], enable_categorical=True)
params = {"max_depth": 7, "learning_rate": 0.015, "tree_method": "hist", "seed": 42, "verbosity": 0}
xgb_model = xgb.train(params, dtrain, num_boost_round=2000, obj=asym_mse_xgb)

MLELog.success("Ensemble models training complete.")


# ==========================================================================================
# --- STAGE 5: Autoregressive Decoder ---
# ==========================================================================================

test_processed["timestamp_abs"] = test_processed["timestamp"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
unique_timestamps = sorted(test_processed["timestamp_abs"].unique())


def ewma(series, alpha=0.3):
    """Calculates Exponentially Weighted Moving Average (EWMA) and returns the last element.
    
    Args:
        series (pd.Series): The pandas Series to average.
        alpha (float): EWMA decay factor. Defaults to 0.3.
        
    Returns:
        float: The final EWMA value of the series.
    """
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]


def summarize_pred(x):
    """Calculates comprehensive summary statistics and quantiles for a set of predictions.
    
    Args:
        x (array-like): Array or Series of predictions.
        
    Returns:
        dict: A dictionary containing statistics like mean, std, min, max, and quantiles.
    """
    x = np.asarray(x, float)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q10": float(np.quantile(x, 0.10)),
        "q25": float(np.quantile(x, 0.25)),
        "q50": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q80": float(np.quantile(x, 0.80)),
        "q85": float(np.quantile(x, 0.85)),
        "q88": float(np.quantile(x, 0.88)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q97": float(np.quantile(x, 0.97)),
        "q98": float(np.quantile(x, 0.98)),
        "q99": float(np.quantile(x, 0.99)),
        "q995": float(np.quantile(x, 0.995)),
        "max": float(x.max()),
    }


def run_cluster_decoder(cb_w, lgb_w, xgb_w, power, cluster_bias, ewma_alpha=0.3):
    """Runs the recursive autoregressive cluster decoder over test intervals.
    
    Iterates chronologically over test timestamps, predicting next-step demand
    using model ensembles, applying cluster biases, and updating lag states.
    
    Args:
        cb_w (float): CatBoost model ensemble weight.
        lgb_w (float): LightGBM model ensemble weight.
        xgb_w (float): XGBoost model ensemble weight.
        power (float): Scaling exponent applied to blended predictions.
        cluster_bias (dict[int, float]): Dictionary mapping cluster IDs to bias constants.
        ewma_alpha (float): Update weight for the EWMA state. Defaults to 0.3.
        
    Returns:
        np.ndarray: Predicted demand array for the test set.
    """
    latest_demand = train_processed.groupby("geohash")["demand"].last().to_dict()
    latest_ewma = train_processed.groupby("geohash")["demand"].apply(ewma).to_dict()
    preds = np.zeros(len(test_processed), dtype=float)

    total_ts = len(unique_timestamps)
    for idx, ts in enumerate(unique_timestamps):
        if idx % 15 == 0 or idx == total_ts - 1:
            MLELog.info(f"Decoding interval {idx+1}/{total_ts} (Timestamp offset: {ts} minutes)")
            
        mask = test_processed["timestamp_abs"] == ts
        if not mask.any():
            continue

        current_ghs = test_processed.loc[mask, "geohash"]

        lags = current_ghs.map(latest_demand).fillna(0.0).values
        ewmas = current_ghs.map(latest_ewma).fillna(0.0).values

        test_processed.loc[mask, "demand_lag1"] = lags
        test_processed_xgb.loc[mask, "demand_lag1"] = lags
        test_processed.loc[mask, "ewma_03"] = ewmas
        test_processed_xgb.loc[mask, "ewma_03"] = ewmas

        batch = test_processed.loc[mask, features]
        batch_xgb = test_processed_xgb.loc[mask, features]

        p_cb = np.clip((1.0 - cb_model1.predict(batch) + 1.0 - cb_model2.predict(batch)) / 2.0, 0.0, 1.0)
        p_lgb = np.clip(lgb_model.predict(batch_xgb), 0.0, 1.0)
        p_xgb = np.clip(xgb_model.predict(xgb.DMatrix(batch_xgb, enable_categorical=True)), 0.0, 1.0)

        blend = cb_w * p_cb + lgb_w * p_lgb + xgb_w * p_xgb

        gh_clusters = current_ghs.map(cluster_map).fillna(0).astype(int).values
        biases = np.array([cluster_bias.get(int(c), cluster_bias.get(0, 0.007)) for c in gh_clusters], dtype=float)

        final = np.clip(np.power(blend, power) + biases, 0.0, 1.0)
        preds[mask] = final

        for gh, pred in zip(current_ghs, final):
            latest_demand[gh] = pred
            latest_ewma[gh] = (1 - ewma_alpha) * latest_ewma.get(gh, 0.0) + ewma_alpha * pred

    return preds


MLELog.success("Final decoder configuration loaded. Autoregressive loop setup complete.")


# ==========================================================================================
# --- STAGE 6: Hyperlocal Blend & V267 Reranking ---
# ==========================================================================================

MLELog.step(5, 5, "Executing Hyperlocal Blending & Reranking")

VERSION = "submission_ready_9229_exact"
OUT_DIR = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

required = ["train_processed", "test_processed", "run_cluster_decoder", "cluster_map"]
missing = [x for x in required if x not in globals()]
if missing:
    raise RuntimeError(f"Missing variables: {missing}. Run the preceding stages first.")


def parse_minute_col(s):
    """Parses various timestamp column formats to extract time in minutes (0 to 1439).
    
    Args:
        s (pd.Series): The series representing raw timestamps.
        
    Returns:
        pd.Series: Integer minute values.
    """
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").astype(float)
        sec = out.notna() & (out > 1440) & (out <= 86400)
        out.loc[sec] = out.loc[sec] / 60.0
    else:
        ss = s.astype(str).str.strip()
        out = pd.to_numeric(ss, errors="coerce").astype(float)
        sec = out.notna() & (out > 1440) & (out <= 86400)
        out.loc[sec] = out.loc[sec] / 60.0
        bad = out.isna()
        if bad.any():
            ex = ss[bad].str.extract(r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{1,2})")
            h = pd.to_numeric(ex["h"], errors="coerce")
            m = pd.to_numeric(ex["m"], errors="coerce")
            ok = h.notna() & m.notna()
            out.loc[bad[bad].index[ok.values]] = (h[ok] * 60 + m[ok]).values
    if out.isna().all():
        out[:] = 0
    return out.fillna(np.nanmedian(out)).round().clip(0, 1439).astype(int)


def prep(df):
    """Prepares and aligns essential columns in a DataFrame for hyperlocal features.
    
    Ensures columns like minute, minutes, hour, cluster, hblock, and slot are created.
    
    Args:
        df (pd.DataFrame): The DataFrame to prepare.
        
    Returns:
        pd.DataFrame: The prepared copy of the DataFrame.
    """
    df = df.copy()
    if "minute" not in df.columns:
        if "minutes" in df.columns:
            df["minute"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0).astype(int)
        elif "timestamp_abs" in df.columns:
            df["minute"] = pd.to_numeric(df["timestamp_abs"], errors="coerce").fillna(0).astype(int) % 1440
        else:
            df["minute"] = parse_minute_col(df["timestamp"])
    if "minutes" not in df.columns:
        df["minutes"] = df["minute"]
    if "hour" not in df.columns:
        df["hour"] = (df["minute"] // 60).astype(int)
    df["geohash"] = df["geohash"].astype(str)
    df["gh5"] = df["geohash"].str[:5]
    df["gh4"] = df["geohash"].str[:4]
    if "cluster" not in df.columns:
        df["cluster"] = df["geohash"].map(cluster_map).fillna(0).astype(int)
    df["hblock"] = np.where(df["minute"] <= 285, 0, np.where(df["minute"] <= 600, 1, 2)).astype(int)
    df["slot"] = (df["minute"] // 15).astype(int)
    return df


def map_group(df, keys, series, default):
    """Maps group stats from a series to a DataFrame based on single or compound keys.
    
    Args:
        df (pd.DataFrame): Target DataFrame.
        keys (list[str]): Column names acting as keys.
        series (pd.Series): Group statistics series.
        default (float): Default fallback value for missing or infinite entries.
        
    Returns:
        np.ndarray: Evaluated mapped stats.
    """
    default = float(default) if np.isfinite(default) else 0.0
    if len(keys) == 1:
        return df[keys[0]].map(series).astype(float).replace([np.inf, -np.inf], np.nan).fillna(default).values
    mp = series.to_dict()
    vals = np.array([mp.get(tuple(x), default) for x in df[keys].values], dtype=float)
    vals[~np.isfinite(vals)] = default
    return vals


def gmean(df, keys):
    """Computes the mean demand grouped by key columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        keys (list[str]): Columns to group by.
        
    Returns:
        pd.Series: Mean demand values indexed by keys.
    """
    return df.groupby(keys)["demand"].mean()


def gcount(df, keys):
    """Computes the count of demand records grouped by key columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        keys (list[str]): Columns to group by.
        
    Returns:
        pd.Series: Counts indexed by keys.
    """
    return df.groupby(keys)["demand"].count()


def pct_rank(a):
    """Computes percentiles for a 1D array of values.
    
    Args:
        a (array-like): Input array.
        
    Returns:
        np.ndarray: Fractional rank values normalized by length.
    """
    return pd.Series(np.asarray(a, float)).rank(method="average").values / len(a)


def corr(a, b):
    """Calculates the Pearson correlation coefficient between two arrays, omitting non-finite entries.
    
    Args:
        a (array-like): First array.
        b (array-like): Second array.
        
    Returns:
        float: Pearson correlation coefficient, or 0.0 if not enough data points or zero variance.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def summarize(x, name):
    """Generates detailed summary metrics and quantiles for audit reports.
    
    Args:
        x (array-like): Data values to summarize.
        name (str): Label/name of the dataset/file.
        
    Returns:
        dict: A dictionary of summary metrics.
    """
    x = np.asarray(x, float)
    out = {
        "file": name,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "rows": int(len(x)),
        "nan": int(np.isnan(x).sum()),
    }
    for q in [0.01,0.05,0.10,0.25,0.50,0.75,0.80,0.85,0.88,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99,0.995]:
        out[f"q{int(q*1000):03d}"] = float(np.quantile(x, q))
    return out


def write_sub(name, pred):
    """Writes demand predictions to a submission file.
    
    Args:
        name (str): File name of the submission.
        pred (np.ndarray): Prediction array.
        
    Returns:
        str: Absolute or relative output path of the file.
    """
    path = os.path.join(OUT_DIR, name)
    pd.DataFrame({
        "Index": test_processed["Index"].values,
        "demand": np.clip(pred, 0, 1)
    }).to_csv(path, index=False)
    return path


def rank_to_values(score, values, group=None, min_n=60):
    """Maps values to scores within groups to match distribution constraints.
    
    Orders indices based on score, sorts corresponding target values, and remaps
    them to preserve overall or group-wise value distribution.
    
    Args:
        score (np.ndarray): Score values defining row ordering.
        values (np.ndarray): Values representing the target distribution.
        group (np.ndarray, optional): Array defining group memberships. Defaults to None.
        min_n (int): Minimum group size required for group-wise mapping. Defaults to 60.
        
    Returns:
        np.ndarray: Sorted target values mapped to input ordering.
    """
    score = np.asarray(score, float)
    values = np.asarray(values, float)
    out = np.full(len(score), np.nan, dtype=float)

    if group is None:
        idx = np.arange(len(score))
        order = idx[np.argsort(score[idx], kind="mergesort")]
        out[order] = np.sort(values[idx])
        return out

    group = np.asarray(group).astype(str)
    for g in pd.unique(group):
        idx = np.where(group == g)[0]
        if len(idx) < min_n:
            continue
        order = idx[np.argsort(score[idx], kind="mergesort")]
        out[order] = np.sort(values[idx])

    miss = ~np.isfinite(out)
    if miss.any():
        idx = np.where(miss)[0]
        order = idx[np.argsort(score[idx], kind="mergesort")]
        out[order] = np.sort(values[idx])

    return out


def make_key(*arrs):
    """Concatenates multiple arrays to form unique compound string keys.
    
    Args:
        *arrs: Varargs of arrays of the same length to join.
        
    Returns:
        np.ndarray: An array of concatenated string keys.
    """
    arrs = [np.asarray(a).astype(str) for a in arrs]
    out = arrs[0].copy()
    for a in arrs[1:]:
        out = np.char.add(np.char.add(out, "_"), a)
    return out


MLELog.info("Building raw AR base...")
base = run_cluster_decoder(
    0.20,
    0.30,
    0.50,
    1.052,
    {0: 0.004, 1: 0.012, 2: 0.008},
    0.30
)
base = np.clip(np.asarray(base, float), 0, 1)

tr = prep(train_processed)
te = prep(test_processed)

if "day" not in tr.columns:
    raise RuntimeError("train_processed needs day column.")

d48 = tr[tr["day"] == 48].copy()
d49 = tr[tr["day"] == 49].copy()
d48e = d48[d48["minute"] <= 120].copy()
d49e = d49[d49["minute"] <= 120].copy()

if len(d48) == 0 or len(d49e) == 0:
    raise RuntimeError("Need day48 full and day49 early rows.")

global_d48 = float(d48["demand"].mean())
global_ratio = float((d49e["demand"].mean() + 1e-6) / (d48e["demand"].mean() + 1e-6))

MLELog.info(f"global_d48: {global_d48:.6f} | global_ratio: {global_ratio:.6f}")


# ------------------------------------------------------------------------------------------
# 2. Hyperlocal direct horizon branch.
# ------------------------------------------------------------------------------------------

MLELog.info("Constructing hyperlocal direct horizon branch...")
profile = np.zeros(len(te), dtype=float)
w_sum = 0.0

profile_specs = [
    (["geohash", "minute"], 0.38),
    (["geohash", "hour"],   0.10),
    (["gh5", "minute"],     0.22),
    (["gh4", "minute"],     0.12),
    (["gh5", "hour"],       0.07),
    (["gh4", "hour"],       0.04),
    (["minute"],            0.05),
    (["hour"],              0.02),
]

for keys, w in profile_specs:
    profile += w * map_group(te, keys, gmean(d48, keys), global_d48)
    w_sum += w

profile = np.clip(profile / max(w_sum, 1e-9), 0, 1)

# Early day49/day48 calibration.
cal = np.zeros(len(te), dtype=float)
cw = 0.0

cal_specs = [
    (["geohash"], 0.34),
    (["gh5"], 0.26),
    (["gh4"], 0.16),
    (["RoadType"], 0.06),
    (["NumberofLanes"], 0.06),
    (["RoadType", "NumberofLanes"], 0.06),
    (["gh5", "RoadType"], 0.03),
    (["gh4", "NumberofLanes"], 0.03),
]

for keys, w in cal_specs:
    a = gmean(d48e, keys)
    b = gmean(d49e, keys)
    ratio = ((b + 1e-5) / (a + 1e-5)).replace([np.inf, -np.inf], np.nan).dropna().clip(0.60, 1.90)
    cal += w * map_group(te, keys, ratio, global_ratio)
    cw += w

cal = np.clip(cal / max(cw, 1e-9), 0.65, 1.80)


def slope_by_group(df, keys):
    """Calculates the linear regression slope of demand vs minutes for group keys.
    
    Args:
        df (pd.DataFrame): Data frame.
        keys (list[str]): Columns to group by.
        
    Returns:
        pd.Series: Slope values indexed by keys.
    """
    rows = []
    for k, g in df.sort_values("minute").groupby(keys):
        if len(g) < 3:
            continue
        x = g["minute"].values.astype(float)
        y = g["demand"].values.astype(float)
        xm = x.mean()
        ym = y.mean()
        den = ((x - xm) ** 2).sum()
        sl = 0.0 if den <= 1e-9 else float(((x - xm) * (y - ym)).sum() / den)
        if not isinstance(k, tuple):
            k = (k,)
        rows.append((*k, sl))
    if len(rows) == 0:
        return pd.Series(dtype=float)
    cols = list(keys) + ["slope"]
    return pd.DataFrame(rows, columns=cols).set_index(keys)["slope"]


global_slope = 0.0
sl_gh = slope_by_group(d49e, ["geohash"])
sl_gh5 = slope_by_group(d49e, ["gh5"])
sl_gh4 = slope_by_group(d49e, ["gh4"])

slope = (
    0.55 * map_group(te, ["geohash"], sl_gh, global_slope)
    + 0.30 * map_group(te, ["gh5"], sl_gh5, global_slope)
    + 0.15 * map_group(te, ["gh4"], sl_gh4, global_slope)
)
slope = np.clip(slope, -0.0035, 0.0035)

last_early = (
    d49e.sort_values("minute")
    .groupby("geohash")["demand"]
    .last()
)
last_gh5 = (
    d49e.sort_values("minute")
    .groupby("gh5")["demand"]
    .last()
)
last_gh4 = (
    d49e.sort_values("minute")
    .groupby("gh4")["demand"]
    .last()
)

last_level = (
    0.55 * map_group(te, ["geohash"], last_early, d49e["demand"].mean())
    + 0.30 * map_group(te, ["gh5"], last_gh5, d49e["demand"].mean())
    + 0.15 * map_group(te, ["gh4"], last_gh4, d49e["demand"].mean())
)

horizon_steps = np.maximum(0, (te["minute"].values - 120) / 15.0)
trend_damp = np.exp(-horizon_steps / 20.0)
trend_forecast = np.clip(last_level + slope * (te["minute"].values - 120) * trend_damp, 0, 1)

cal_profile = np.clip(profile * cal, 0, 1)

# Combine hyperlocal value forecasts.
local_value = np.clip(
    0.68 * cal_profile
    + 0.22 * trend_forecast
    + 0.10 * profile,
    0,
    1
)

# Confidence: more weight where geohash has real history and stable early calibration.
cnt_gh_d48 = map_group(te, ["geohash"], gcount(d48, ["geohash"]), 0)
cnt_gh_d49e = map_group(te, ["geohash"], gcount(d49e, ["geohash"]), 0)
cnt_gh5_d49e = map_group(te, ["gh5"], gcount(d49e, ["gh5"]), 0)

conf = (
    0.50 * np.clip(cnt_gh_d48 / 50.0, 0, 1)
    + 0.35 * np.clip(cnt_gh_d49e / 5.0, 0, 1)
    + 0.15 * np.clip(cnt_gh5_d49e / 25.0, 0, 1)
)
conf = np.clip(conf, 0, 1)

# Less local weight very late unless local confidence is high.
late = te["minute"].values > 600
conf_late = conf.copy()
conf_late[late] *= 0.75


# ------------------------------------------------------------------------------------------
# 3. Convert local branch safely.
# ------------------------------------------------------------------------------------------

MLELog.info("Preserving prediction distribution bounds and residuals...")
cluster = te["cluster"].values
hblock = te["hblock"].values
hour = te["hour"].values
g_ch = make_key(cluster, hblock)
g_chh = make_key(cluster, hblock, hour)

local_score = (
    0.42 * pct_rank(local_value)
    + 0.28 * pct_rank(cal_profile)
    + 0.15 * pct_rank(trend_forecast)
    + 0.10 * pct_rank(conf)
    + 0.05 * pct_rank(profile)
)

# Value mapping: use local score for row order, but preserve base distribution by group.
local_rankmap_ch = rank_to_values(local_score, base, g_ch, min_n=80)
local_rankmap_chh = rank_to_values(local_score, base, g_chh, min_n=35)

# Local residual relative to base.
resid = local_value - base
resid = np.clip(resid, -0.18, 0.18)

base_rank = pct_rank(base)
mid_band = (base_rank >= 0.35) & (base_rank <= 0.90)
shelf_band = (base_rank >= 0.80) & (base_rank <= 0.97)
safe_band = (base_rank >= 0.20) & (base_rank <= 0.97)
protect_top = base_rank > 0.97


def blend_value(name, w, conf_arr, mask):
    """Blends base predictions with local value predictions based on confidence and mask conditions.
    
    Args:
        name (str): Identifier name.
        w (float): Local value blending weight.
        conf_arr (np.ndarray): Local confidence levels.
        mask (np.ndarray): Boolean mask indicating where to apply the blend.
        
    Returns:
        np.ndarray: Blended prediction values.
    """
    out = base.copy()
    ww = w * conf_arr
    out[mask] = (1 - ww[mask]) * base[mask] + ww[mask] * local_value[mask]
    out[protect_top] = base[protect_top]
    return np.clip(out, 0, 1)


def blend_rankmap(name, alt, w, conf_arr, mask):
    """Blends base predictions with rankmap values based on confidence and mask conditions.
    
    Args:
        name (str): Identifier name.
        alt (np.ndarray): Alternative target values.
        w (float): Blending weight.
        conf_arr (np.ndarray): Confidence array.
        mask (np.ndarray): Boolean mask indicating where to apply the blend.
        
    Returns:
        np.ndarray: Blended prediction values.
    """
    out = base.copy()
    ww = w * conf_arr
    out[mask] = (1 - ww[mask]) * base[mask] + ww[mask] * alt[mask]
    out[protect_top] = base[protect_top]
    return np.clip(out, 0, 1)


def add_residual(name, alpha, conf_arr, mask):
    """Adds local residuals to base predictions scaled by a factor and confidence levels.
    
    Args:
        name (str): Identifier name.
        alpha (float): Scaling factor for the residual.
        conf_arr (np.ndarray): Confidence values.
        mask (np.ndarray): Boolean mask indicating where to apply the residual.
        
    Returns:
        np.ndarray: Adjusted predictions.
    """
    out = base.copy()
    delta = alpha * conf_arr * resid
    out[mask] = base[mask] + delta[mask]
    out[protect_top] = base[protect_top]
    return np.clip(out, 0, 1)


candidates = {
    # Rankmap branch: should preserve distribution best.
    "submission_v266_local_rankmap_ch_w08.csv":
        blend_rankmap("r1", local_rankmap_ch, 0.08, conf_late, safe_band),

    "submission_v266_local_rankmap_ch_w14.csv":
        blend_rankmap("r2", local_rankmap_ch, 0.14, conf_late, safe_band),

    "submission_v266_local_rankmap_chh_w10.csv":
        blend_rankmap("r3", local_rankmap_chh, 0.10, conf_late, safe_band),

    # Direct local value branch: bigger but riskier.
    "submission_v266_local_value_mid_w06.csv":
        blend_value("v1", 0.06, conf_late, mid_band),

    "submission_v266_local_value_mid_w10.csv":
        blend_value("v2", 0.10, conf_late, mid_band),

    "submission_v266_local_value_shelf_w06.csv":
        blend_value("v3", 0.06, conf_late, shelf_band),

    # Residual branch: tries to capture real local correction.
    "submission_v266_local_resid_safe_a025.csv":
        add_residual("e1", 0.025, conf_late, safe_band),

    "submission_v266_local_resid_main_a040.csv":
        add_residual("e2", 0.040, conf_late, safe_band),

    "submission_v266_local_resid_shelf_a050.csv":
        add_residual("e3", 0.050, conf_late, shelf_band),

    # Ensemble of safe branches.
    "submission_v266_local_combo_safe.csv":
        np.clip(
            0.50 * blend_rankmap("r1", local_rankmap_ch, 0.08, conf_late, safe_band)
            + 0.30 * add_residual("e1", 0.025, conf_late, safe_band)
            + 0.20 * blend_value("v1", 0.06, conf_late, mid_band),
            0,
            1
        ),
}


def audit(name, pred):
    """Runs a diagnostic audit on predictions compared to raw base predictions.
    
    Args:
        name (str): The label for this prediction run.
        pred (np.ndarray): Prediction array.
        
    Returns:
        dict: Evaluation results, metrics, and percentiles comparison.
    """
    pred = np.asarray(pred, float)
    d = pred - base
    minute = te["minute"].values

    top10 = base >= np.quantile(base, .90)
    top05 = base >= np.quantile(base, .95)
    top02 = base >= np.quantile(base, .98)
    top01 = base >= np.quantile(base, .99)

    a = summarize(pred, name)
    a.update({
        "corr_base": corr(pred, base),
        "rank_corr_base": corr(pct_rank(pred), pct_rank(base)),
        "corr_local_value": corr(pred, local_value),
        "rmse_base": float(np.sqrt(np.mean(d*d))),
        "mae_base": float(np.mean(np.abs(d))),
        "rows_abs_gt_0005": int((np.abs(d) > 0.0005).sum()),
        "rows_abs_gt_001": int((np.abs(d) > 0.001).sum()),
        "rows_abs_gt_002": int((np.abs(d) > 0.002).sum()),
        "max_abs_delta": float(np.max(np.abs(d))),
        "q90_delta": float(np.quantile(pred, .90) - np.quantile(base, .90)),
        "q95_delta": float(np.quantile(pred, .95) - np.quantile(base, .95)),
        "q97_delta": float(np.quantile(pred, .97) - np.quantile(base, .97)),
        "q98_delta": float(np.quantile(pred, .98) - np.quantile(base, .98)),
        "q99_delta": float(np.quantile(pred, .99) - np.quantile(base, .99)),
        "top10_delta": float(d[top10].mean()),
        "top05_delta": float(d[top05].mean()),
        "top02_delta": float(d[top02].mean()),
        "top01_delta": float(d[top01].mean()),
        "early_delta": float(d[minute <= 285].mean()),
        "mid_delta": float(d[(minute > 285) & (minute <= 600)].mean()),
        "late_delta": float(d[minute > 600].mean()),
    })
    return a


base_summary = summarize(base, "base_92_18_raw_ar")
local_summary = summarize(local_value, "hyperlocal_value")


# ------------------------------------------------------------------------------------------
# 5. V267 overlay: dynamic neighbor correction + MoE rerank + tail calibration.
# ------------------------------------------------------------------------------------------

MLELog.info("Building V267 dynamic neighbor / MoE overlay...")

base_9229 = np.clip(np.asarray(candidates["submission_v266_local_combo_safe.csv"], float), 0, 1)
base_rank_9229 = pct_rank(base_9229)

safe_band_267 = (base_rank_9229 >= 0.20) & (base_rank_9229 <= 0.975)
shelf_band_267 = (base_rank_9229 >= 0.80) & (base_rank_9229 <= 0.975)
protect_top_267 = base_rank_9229 > 0.985

minute_267 = te["minute"].values.astype(int)
cluster_267 = te["cluster"].values.astype(int) if "cluster" in te.columns else np.zeros(len(te), dtype=int)
hblock_267 = te["hblock"].values.astype(int) if "hblock" in te.columns else np.where(minute_267 <= 285, 0, np.where(minute_267 <= 600, 1, 2))
hour_267 = te["hour"].values.astype(int) if "hour" in te.columns else minute_267 // 60
road_267 = te["RoadType"].astype(str).values if "RoadType" in te.columns else np.array(["Missing"] * len(te))

coord_cols = None
for c1, c2 in [("latitude", "longitude"), ("lat", "lon")]:
    if c1 in te.columns and c2 in te.columns:
        coord_cols = (c1, c2)
        break
if coord_cols is None:
    raise RuntimeError("V267 overlay needs latitude/longitude columns in the processed test frame.")

geo_unique = te[["geohash", coord_cols[0], coord_cols[1]]].drop_duplicates("geohash").reset_index(drop=True)
coords_267 = geo_unique[[coord_cols[0], coord_cols[1]]].values
nn_267 = NearestNeighbors(n_neighbors=min(6, len(geo_unique)), metric="euclidean")
nn_267.fit(coords_267)
_, idxs_267 = nn_267.kneighbors(coords_267)

ghs_267 = geo_unique["geohash"].astype(str).values
neighbor_map_267 = {}
for i, gh in enumerate(ghs_267):
    neighs = [ghs_267[j] for j in idxs_267[i][1:] if j < len(ghs_267)]
    neighbor_map_267[gh] = neighs[:5]

lookup_267 = {}
for gh, m, p in zip(te["geohash"].astype(str).values, minute_267, base_9229):
    lookup_267[(gh, int(m))] = float(p)

neigh_mean_267 = np.zeros(len(te), dtype=float)
neigh_cnt_267 = np.zeros(len(te), dtype=float)
for i, (gh, m) in enumerate(zip(te["geohash"].astype(str).values, minute_267)):
    vals = []
    for ng in neighbor_map_267.get(gh, []):
        v = lookup_267.get((ng, int(m)), np.nan)
        if np.isfinite(v):
            vals.append(v)
    if vals:
        neigh_mean_267[i] = float(np.mean(vals))
        neigh_cnt_267[i] = len(vals)
    else:
        neigh_mean_267[i] = base_9229[i]
        neigh_cnt_267[i] = 0

spatial_resid_267 = np.clip(neigh_mean_267 - base_9229, -0.08, 0.08)
spatial_conf_267 = np.clip(neigh_cnt_267 / 5.0, 0, 1)
spatial_score_267 = pct_rank(neigh_mean_267)

horizon_gate_267 = np.where(minute_267 <= 285, 0.85, np.where(minute_267 <= 600, 1.00, 0.75))
dyn_gate_267 = conf_late * spatial_conf_267 * horizon_gate_267
dyn_gate_267[protect_top_267] = 0.0

rerank_score_267 = (
    0.50 * pct_rank(base_9229)
    + 0.28 * local_score
    + 0.17 * spatial_score_267
    + 0.05 * pct_rank(conf)
)

g_chr_267 = make_key(cluster_267, hblock_267, road_267)
rank_chr_267 = rank_to_values(rerank_score_267, base_9229, g_chr_267, min_n=50)


def blend_rank_v267(alt, w, mask):
    """Blends the intermediate base with alternative rank values for V267.
    
    Args:
        alt (np.ndarray): Alternative rank values.
        w (float): Blending weight.
        mask (np.ndarray): Boolean mask indicating where to blend.
        
    Returns:
        np.ndarray: Blended predictions.
    """
    out = base_9229.copy()
    ww = w * conf_late
    out[mask] = (1 - ww[mask]) * base_9229[mask] + ww[mask] * alt[mask]
    out[protect_top_267] = base_9229[protect_top_267]
    return np.clip(out, 0, 1)


def add_dyn_v267(alpha, mask):
    """Applies dynamic neighbor spatial residue adjustment to the intermediate base for V267.
    
    Args:
        alpha (float): Adjustment factor.
        mask (np.ndarray): Boolean mask.
        
    Returns:
        np.ndarray: Adjusted predictions.
    """
    out = base_9229.copy()
    delta = alpha * dyn_gate_267 * spatial_resid_267
    out[mask] = base_9229[mask] + delta[mask]
    out[protect_top_267] = base_9229[protect_top_267]
    return np.clip(out, 0, 1)


tail_strength_267 = np.clip((base_rank_9229 - 0.86) / 0.13, 0, 1) ** 1.5
cluster1_267 = cluster_267 == 1
tail_c1_up_267 = base_9229.copy()
tail_mask_267 = shelf_band_267 & cluster1_267
tail_c1_up_267[tail_mask_267] += 0.0045 * tail_strength_267[tail_mask_267] * conf_late[tail_mask_267]
tail_c1_up_267[protect_top_267] = base_9229[protect_top_267]
tail_c1_up_267 = np.clip(tail_c1_up_267, 0, 1)

moe_rank_chr_w07_267 = blend_rank_v267(rank_chr_267, 0.07, safe_band_267)
dyn_neighbor_a020_267 = add_dyn_v267(0.020, safe_band_267)
v267_combo_safe = np.clip(
    0.50 * moe_rank_chr_w07_267
    + 0.30 * dyn_neighbor_a020_267
    + 0.20 * tail_c1_up_267,
    0,
    1,
)


def audit_vs_reference(name, pred, ref):
    """Audits the differences and similarities between predictions and reference predictions.
    
    Args:
        name (str): Label for the comparison audit.
        pred (np.ndarray): The prediction to evaluate.
        ref (np.ndarray): Reference prediction.
        
    Returns:
        dict: A dictionary of differences and correlation metrics.
    """
    pred = np.asarray(pred, float)
    ref = np.asarray(ref, float)
    d = pred - ref
    return {
        "name": name,
        "corr_ref": corr(pred, ref),
        "mae_delta": float(np.mean(np.abs(d))),
        "rmse_delta": float(np.sqrt(np.mean(d * d))),
        "max_abs_delta": float(np.max(np.abs(d))),
        "rows_abs_gt_001": int((np.abs(d) > 0.001).sum()),
        "rows_abs_gt_002": int((np.abs(d) > 0.002).sum()),
        "q90_delta": float(np.quantile(pred, 0.90) - np.quantile(ref, 0.90)),
        "q95_delta": float(np.quantile(pred, 0.95) - np.quantile(ref, 0.95)),
        "q98_delta": float(np.quantile(pred, 0.98) - np.quantile(ref, 0.98)),
        "q99_delta": float(np.quantile(pred, 0.99) - np.quantile(ref, 0.99)),
    }


v267_audit_vs_9229 = audit_vs_reference("v267_combo_safe_vs_v266_9229", v267_combo_safe, base_9229)

final_name = FINAL_SUBMISSION_NAME
final_pred = np.clip(np.asarray(v267_combo_safe, float), 0, 1)
final_path = os.path.join(OUT_DIR, final_name)
pd.DataFrame({
    "Index": test_processed["Index"].values,
    "demand": final_pred,
}).to_csv(final_path, index=False)

final_audit = audit(final_name, final_pred)

# Validation against sample submission when available.
if SAMPLE_PATH.exists():
    sample = pd.read_csv(SAMPLE_PATH)
    if len(sample) != len(final_pred):
        raise ValueError(f"sample_submission rows {len(sample)} != prediction rows {len(final_pred)}")
    if "Index" in sample.columns and not pd.Series(test_processed["Index"].values).equals(sample["Index"]):
        raise ValueError("Test Index order does not match sample_submission.csv")

if len(final_pred) != len(test_processed):
    raise ValueError("Prediction row count mismatch")
if not np.isfinite(final_pred).all():
    raise ValueError("Prediction contains NaN/inf")
if final_pred.min() < -1e-12 or final_pred.max() > 1 + 1e-12:
    raise ValueError("Prediction outside [0, 1]")

# Reviewer-friendly charts.
chart_dir = Path(OUT_DIR) / "submission_charts"
chart_dir.mkdir(parents=True, exist_ok=True)

try:
    plt.figure(figsize=(10, 5))
    plt.hist(base, bins=80, alpha=0.45, label="raw AR base")
    plt.hist(base_9229, bins=80, alpha=0.35, label="V266 intermediate")
    plt.hist(final_pred, bins=80, alpha=0.45, label="final submission")
    plt.xlabel("demand")
    plt.ylabel("row count")
    plt.title("Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "prediction_distribution.png", dpi=160)
    plt.close()

    qs = np.linspace(0.01, 0.995, 120)
    plt.figure(figsize=(10, 5))
    plt.plot(qs, np.quantile(base, qs), label="raw AR base")
    plt.plot(qs, np.quantile(base_9229, qs), label="V266 intermediate")
    plt.plot(qs, np.quantile(final_pred, qs), label="final submission")
    plt.xlabel("quantile")
    plt.ylabel("demand")
    plt.title("Quantile Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / "quantile_curve.png", dpi=160)
    plt.close()

    delta = final_pred - base
    minute_vals = te["minute"].values
    block_names = ["early <=285", "mid 286-600", "late >600"]
    block_masks = [minute_vals <= 285, (minute_vals > 285) & (minute_vals <= 600), minute_vals > 600]
    block_means = [float(delta[m].mean()) for m in block_masks]
    plt.figure(figsize=(7, 4))
    plt.bar(block_names, block_means)
    plt.ylabel("mean final - raw base")
    plt.title("Correction by Horizon Block")
    plt.tight_layout()
    plt.savefig(chart_dir / "correction_by_horizon.png", dpi=160)
    plt.close()

    MLELog.success(f"Reviewer-friendly charts generated in: {chart_dir}")
except Exception as chart_error:
    MLELog.warning(f"Chart generation skipped: {repr(chart_error)}")

MLELog.print_table("Base Raw AR Prediction Summary", base_summary)
MLELog.print_table("Hyperlocal Value Prediction Summary", local_summary)
MLELog.print_table("Final Submission Audit vs Raw Base", final_audit)
MLELog.print_table("V267 Audit vs V266 Intermediate Base", v267_audit_vs_9229)

MLELog.info("Wrote only one submission file:")
MLELog.info(final_path)

MLELog.success(f"Final submission generated at {final_path}")
MLELog.header("SUBMISSION PIPELINE COMPLETED")
