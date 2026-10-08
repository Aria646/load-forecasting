"""
Function: Validate model generalization on other regions
Project: Energy Data Analysis - Short Term Load Forecasting (STLF)

Load saved models and evaluate on other regions (QLD1, SA1, TAS1, VIC1)
Metrics: MAE, MAPE, RMSE, R2
"""

import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

warnings.filterwarnings('ignore')

# ============================================================
# Helper Functions (match training logic exactly)
# ============================================================

def create_horizon_targets(y_base: np.ndarray, horizon: int) -> np.ndarray:
    """
    Create multi-horizon targets from 1D base array.
    Matches training's create_horizon_targets exactly.
    
    Args:
        y_base: 1D array of target values (n_samples,)
        horizon: Prediction horizon (number of steps ahead)
    
    Returns:
        2D array of shape (n_samples, horizon) with NaN padding at end.
        Column order: [t+horizon, t+horizon-1, ..., t+1] (REVERSE chronological order).
        Note: Last (horizon-1) rows contain NaN and should be dropped before training.
        To get chronological order [t+1, ..., t+horizon], use targets[:, ::-1] or slice [:, -horizon:].
    """
    n = len(y_base)
    targets = np.zeros((n, horizon))
    for i in range(horizon):
        # Column i corresponds to t+horizon-i (reverse order: i=0 -> t+h, i=h-1 -> t+1)
        targets[:-horizon+i, i] = y_base[horizon-i:] if horizon-i > 0 else y_base[:]
        if horizon-i > 0:
            targets[-(horizon-i):, i] = np.nan
    return targets


def is_rnn_model(model_name: str) -> bool:
    """Check if model is RNN variant (LSTM/GRU/BiLSTM/BiGRU)."""
    name_lower = model_name.lower()
    return any(v in name_lower for v in ('lstm', 'gru', 'bilstm', 'bigru'))


def align_targets(
    model,
    X_test_base: np.ndarray,
    X_test: np.ndarray,
    y_test_base: np.ndarray,
    y_test_horizon: np.ndarray,
    horizon: int,
    lookback: int,
    is_rnn: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align predictions and targets per model type.
    Returns (y_pred, y_true_aligned) both as 2D arrays.
    
    RNN: predict on full X_test_base, align target via lookback slice, force n_horizons=1
    Tree/ANN: predict on filtered X_test, use per-horizon target, n_horizons=horizon
    """
    if is_rnn:
        # RNN: predict on full test base
        y_pred = model.predict(X_test_base)
        # Align target: skip lookback samples
        y_true = y_test_base[lookback:lookback + len(y_pred)]
        # Force 2D with single horizon (training Evaluator sees (n,1) -> n_horizons=1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        n_horizons = 1
    else:
        # Tree/ANN: predict on filtered test, use per-horizon target
        y_pred = model.predict(X_test)
        y_true = y_test_horizon
        n_horizons = horizon
    
    # Ensure y_pred is 2D
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    return y_pred, y_true, n_horizons


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_horizons: int,
    model_name: str,
    horizon: int,
    region: str
) -> dict:
    """
    Compute metrics matching training's Evaluator.calculate_metrics exactly.
    y_true: (n_samples, n_horizons) or (n_samples,) -> reshaped to (n, n_horizons)
    y_pred: (n_samples, n_horizons) or (n_samples,) -> reshaped to (n, n_horizons)
    n_horizons: number of horizon steps to evaluate (1 for RNN, horizon for Tree/ANN)
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    mae_per_h, mape_per_h, rmse_per_h, r2_per_h = [], [], [], []

    for h in range(n_horizons):
        yt = y_true[:, h]
        # For RNN: y_pred has shape (n, horizon) but n_horizons=1, so take first column
        yp = y_pred[:, h] if y_pred.shape[1] > h else y_pred[:, 0]
        
        mae = mean_absolute_error(yt, yp)
        yt_safe = np.where(yt == 0, 1e-6, yt)
        mape = np.mean(np.abs((yt - yp) / yt_safe)) * 100
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp) * 100  # Match training: R2 as percentage
        
        mae_per_h.append(mae)
        mape_per_h.append(mape)
        rmse_per_h.append(rmse)
        r2_per_h.append(r2)

    return {
        'Region': region,
        'Model': model_name,
        'Horizon': horizon,
        'MAE': round(np.mean(mae_per_h), 2),
        'MAPE': round(np.mean(mape_per_h), 2),
        'RMSE': round(np.mean(rmse_per_h), 2),
        'R2': round(np.mean(r2_per_h), 2),
    }


# ============================================================
# Configuration
# ============================================================

class Config:
    def __init__(self, horizons=None, models=None, region=None, test_year=None):
        self.horizons = horizons or [1, 24, 168]
        self.model_names = models or ['xgboost', 'lightgbm', 'lstm', 'gru', 'bilstm', 'bigru']
        self.region = region or 'NSW1'
        self.test_year = test_year or YEARS[-1]
        self.lookback = 168
        self.batch_size = 32
        self.hidden_size = 128
        self.num_layers = 2
        self.lr = 0.001
        self.output_dir = Path('./results/validation')
        self.model_dir = Path('./results/models')
        self.data_file = f'aemo_features_{YEARS[0]}_{YEARS[-1]}.csv'
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Base Classes
# ============================================================

class ModelBase(ABC):
    """Abstract base class for all models"""

    def __init__(self, name: str, horizon: int, config: 'Config'):
        self.name = name
        self.horizon = horizon
        self.config = config
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, feature_cols):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass

    def save(self, path: Path):
        """Save model to disk"""
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'horizon': self.horizon,
            'name': self.name,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"  [SAVE] Model saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        instance = cls(save_dict['name'], save_dict['horizon'], None)
        instance.model = save_dict['model']
        instance.feature_scaler = save_dict['feature_scaler']
        instance.target_scaler = save_dict['target_scaler']
        return instance

class HybridRNN(nn.Module):
    """Hybrid RNN: RNN output + explicit lag_168 feature concatenation"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, variant, lag_168_idx):
        super().__init__()
        self.variant = variant
        self.lag_168_idx = lag_168_idx
        rnn_hidden = hidden_size

        if variant == 'lstm':
            self.rnn = nn.LSTM(input_size, rnn_hidden, num_layers, batch_first=True, dropout=0.1)
        elif variant == 'gru':
            self.rnn = nn.GRU(input_size, rnn_hidden, num_layers, batch_first=True, dropout=0.1)
        elif variant == 'bilstm':
            self.rnn = nn.LSTM(input_size, rnn_hidden, num_layers, batch_first=True,
                              bidirectional=True, dropout=0.1)
            rnn_hidden *= 2
        elif variant == 'bigru':
            self.rnn = nn.GRU(input_size, rnn_hidden, num_layers, batch_first=True,
                             bidirectional=True, dropout=0.1)
            rnn_hidden *= 2
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.fc1 = nn.Linear(rnn_hidden + 1, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lag_168_feat = x[:, 0, self.lag_168_idx:self.lag_168_idx+1] if self.lag_168_idx >= 0 else torch.zeros(x.size(0), 1, device=x.device)
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        combined = torch.cat([last_hidden, lag_168_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ANN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ============================================================
# LSTM Model Wrapper
# ============================================================

class LSTMModel(ModelBase):
    """LSTM/GRU/BiLSTM/BiGRU with hybrid architecture (RNN output + explicit lag_168)"""

    def __init__(self, name: str, horizon: int, config: Config, variant: str = 'lstm'):
        super().__init__(name, horizon, config)
        self.variant = variant  # 'lstm', 'gru', 'bilstm', 'bigru'

    def predict(self, X):
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        # Create sequences for prediction (need last lookback points for each prediction)
        processor = DataProcessor(self.config)
        X_seq, _ = processor.create_sequences(X_scaled, np.zeros(len(X_scaled)), self.config.lookback, self.horizon)

        X_t = torch.FloatTensor(X_seq).to(self.config.device)
        test_loader = DataLoader(TensorDataset(X_t), batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for bx in test_loader:
                bx = bx[0].to(self.config.device) if isinstance(bx, (list, tuple)) else bx.to(self.config.device)
                pred = self.model(bx)
                preds.append(pred.cpu().numpy())

        y_pred_scaled = np.vstack(preds)  # Shape: (n_samples, horizon)
        # Clip to valid range before inverse transform
        y_pred_scaled = np.clip(y_pred_scaled, -1.0, 1.0)
        # Reshape for inverse_transform (scaler was fit on flattened 1D data)
        orig_shape = y_pred_scaled.shape
        y_pred_scaled_flat = y_pred_scaled.reshape(-1, 1)
        y_pred_flat = self.target_scaler.inverse_transform(y_pred_scaled_flat)
        y_pred = y_pred_flat.reshape(orig_shape)
        return y_pred

    def set_test_data(self, X_test):
        self.test_X = X_test


# ============================================================
# Configuration
# ============================================================

class Config:
    def __init__(self, horizons=None, models=None, region=None, test_year=None):
        self.horizons = horizons or [1, 24, 168]
        self.model_names = models or ['xgboost', 'lightgbm', 'lstm', 'gru', 'bilstm', 'bigru']
        self.region = region or 'NSW1'
        self.test_year = test_year or YEARS[-1]
        self.lookback = 168
        self.batch_size = 32
        self.hidden_size = 128
        self.num_layers = 2
        self.lr = 0.001
        self.output_dir = Path('./results/validation')
        self.model_dir = Path('./results/models')
        self.data_file = f'aemo_features_{YEARS[0]}_{YEARS[-1]}.csv'
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Cross-region adaptation options
        self.adapt_target_scaler = False  # Use region-specific target scaling (default OFF)
        self.calibration_ratio = 0.1  # Use 10% of test data for calibration


# ============================================================
# Data Processing
# ============================================================

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.feature_cols = None

    def load_and_prepare(self, region):
        """Load data for specific region"""
        print(f"\n[DATA] Loading data for region: {region}")

        df = pd.read_csv(self.config.data_file)
        df['HOUR'] = pd.to_datetime(df['HOUR'])

        # Select region
        region_df = df[df['REGIONID'] == region].copy()
        print(f"  [INFO] Region {region}: {len(region_df):,} records")

        # Drop columns with severe missing values
        drop_cols = ['VISIBILITY', 'PRECIPITATION', 'WIND_SPEED', 'REGIONID']
        region_df = region_df.drop(columns=[c for c in drop_cols if c in region_df.columns])

        # Fill minor missing weather data
        weather_cols = ['TEMPERATURE', 'DEW_POINT', 'HUMIDITY', 'PRESSURE']
        for c in weather_cols:
            if c in region_df.columns:
                region_df[c] = region_df[c].fillna(region_df[c].median())

        region_df = region_df.fillna(0)
        region_df = region_df.sort_values('HOUR').reset_index(drop=True)

        # Test set: use configured test year
        test_year = getattr(self.config, 'test_year', YEARS[-1])
        test_df = region_df[region_df['HOUR'].dt.year == test_year].copy()
        print(f"  [INFO] Test ({test_year}): {len(test_df):,} records")

        # Feature columns
        exclude_cols = ['HOUR', 'TOTALDEMAND']
        feature_cols = [c for c in region_df.columns if c not in exclude_cols]
        self.feature_cols = feature_cols

        return test_df, feature_cols

    @staticmethod
    def create_sequences(X, y, lookback, horizon):
        X_seq, y_seq = [], []
        for i in range(lookback, len(X) - horizon + 1):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i:i+horizon])
        return np.array(X_seq), np.array(y_seq)


class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.results = []

    @staticmethod
    def calculate_metrics(y_true, y_pred, name, horizon, region):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        n_horizons = y_true.shape[1]
        mae_per_h, mape_per_h, rmse_per_h, r2_per_h = [], [], [], []

        for h in range(n_horizons):
            yt = y_true[:, h]
            yp = y_pred[:, h]
            mae = mean_absolute_error(yt, yp)
            yt_safe = np.where(yt == 0, 1e-6, yt)
            mape = np.mean(np.abs((yt - yp) / yt_safe)) * 100
            rmse = np.sqrt(mean_squared_error(yt, yp))
            r2 = r2_score(yt, yp)
            mae_per_h.append(mae)
            mape_per_h.append(mape)
            rmse_per_h.append(rmse)
            r2_per_h.append(r2)

        return {
            'Region': region,
            'Model': name,
            'Horizon': horizon,
            'MAE': np.mean(mae_per_h),
            'MAPE': np.mean(mape_per_h),
            'RMSE': np.mean(rmse_per_h),
            'R2': np.mean(r2_per_h) * 100,
        }

    def add_result(self, metrics):
        self.results.append(metrics)

    def get_results_df(self):
        df = pd.DataFrame(self.results)
        return df.round(4)

    def save_report(self, output_dir: Path):
        df = pd.DataFrame(self.results)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / 'validation_metrics.csv', index=False)

        with open(output_dir / 'validation_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Cross-Region Generalization Validation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")

            # Summary per region
            f.write("SUMMARY BY REGION:\n")
            f.write("-" * 80 + "\n")
            for reg in df['Region'].unique():
                reg_df = df[df['Region'] == reg]
                for h in sorted(df['Horizon'].unique()):
                    h_df = reg_df[reg_df['Horizon'] == h]
                    if len(h_df) > 0:
                        best = h_df.loc[h_df['MAPE'].idxmin()]
                        f.write(f"  {reg} {h}h: Best={best['Model']} (MAPE={best['MAPE']:.2f}%)\n")


# ============================================================
# Model Loading & Prediction
# ============================================================

def load_model(model_path: Path, config, feature_cols, horizon, variant=None):
    """Load saved model from pickle file"""
    with open(model_path, 'rb') as f:
        save_dict = pickle.load(f)

    name = save_dict['name']
    model_type = name.lower()

    if 'xgboost' in model_type.lower():
        # XGBoost - list of models per horizon step
        model = type('XGBWrapper', (), {})()
        model.models = save_dict['model']
        model.feature_scaler = save_dict['feature_scaler']
        model.target_scaler = save_dict['target_scaler']
        model.horizon = save_dict['horizon']
        model.name = save_dict['name']

        def predict(X):
            X_scaled = model.feature_scaler.transform(X)
            preds = np.column_stack([m.predict(X_scaled) for m in model.models])
            preds = model.target_scaler.inverse_transform(preds)
            return preds
        model.predict = predict
        return model

    elif 'lightgbm' in model_type.lower() or 'lgb' in model_type.lower():
        model = type('LGBWrapper', (), {})()
        model.models = save_dict['model']
        model.feature_scaler = save_dict['feature_scaler']
        model.target_scaler = save_dict['target_scaler']
        model.horizon = save_dict['horizon']
        model.name = save_dict['name']

        def predict(X):
            X_scaled = model.feature_scaler.transform(X)
            preds = np.column_stack([m.predict(X_scaled) for m in model.models])
            preds = model.target_scaler.inverse_transform(preds)
            return preds
        model.predict = predict
        return model

    elif 'ann' in model_type.lower():
        model = type('ANNWrapper', (), {})()
        model.model = save_dict['model']
        model.feature_scaler = save_dict['feature_scaler']
        model.target_scaler = save_dict['target_scaler']
        model.horizon = save_dict['horizon']
        model.name = save_dict['name']
        model.config = type('Config', (), {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')})()

        def predict(X):
            X_scaled = model.feature_scaler.transform(X)
            X_t = torch.FloatTensor(X_scaled).to(model.config.device)
            model.model.eval()
            with torch.no_grad():
                pred = model.model(torch.FloatTensor(X_scaled).to(model.config.device)).cpu().numpy()
            pred = model.target_scaler.inverse_transform(pred)
            return pred
        model.predict = predict
        return model

    else:
        # LSTM/GRU/BiLSTM/BiGRU
        model = type('RNNWrapper', (), {})()
        model.model = save_dict['model']
        model.feature_scaler = save_dict['feature_scaler']
        model.target_scaler = save_dict['target_scaler']
        model.horizon = save_dict['horizon']
        model.name = save_dict['name']
        model.config = type('Config', (), {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': 32,
            'lookback': 168
        })()

        def predict(X):
            from torch.utils.data import DataLoader, TensorDataset
            X_scaled = model.feature_scaler.transform(X)
            processor = DataProcessor(type('Config', (), {'lookback': 168})())
            X_seq, _ = processor.create_sequences(X_scaled, np.zeros(len(X_scaled)), 168, model.horizon)
            X_t = torch.FloatTensor(X_seq).to(model.config.device)
            test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_seq)), batch_size=32, shuffle=False)
            model.model.eval()
            preds = []
            with torch.no_grad():
                for bx in test_loader:
                    bx = bx[0].to(model.config.device) if isinstance(bx, (list, tuple)) else bx.to(model.config.device)
                    pred = model.model(bx)
                    preds.append(pred.cpu().numpy())
            y_pred_scaled = np.vstack(preds)
            y_pred_scaled = np.clip(y_pred_scaled, -1.0, 1.0)
            orig_shape = y_pred_scaled.shape
            y_pred_scaled_flat = y_pred_scaled.reshape(-1, 1)
            y_pred_flat = model.target_scaler.inverse_transform(y_pred_scaled)
            y_pred = y_pred_flat.reshape(orig_shape)
            return y_pred
        model.predict = predict
        return model


def evaluate_region(config, region, test_df, feature_cols, evaluator):
    """Evaluate all models on a specific region"""
    print(f"\n{'='*60}")
    print(f"VALIDATING REGION: {region}")
    print(f"{'='*60}")

    X_test_base = test_df[feature_cols].fillna(0).values
    y_test_base = test_df['TOTALDEMAND'].values
    
    print(f"  [INFO] Target stats: mean={y_test_base.mean():.1f}, std={y_test_base.std():.1f}, range=[{y_test_base.min():.1f}, {y_test_base.max():.1f}]")

    # Reusable config for loaded models
    model_config = type('Config', (), {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'lookback': 168
    })()

    for horizon in config.horizons:
        print(f"\n  Horizon: {horizon}h")

        # Create targets for THIS horizon
        y_test_tree_2d = create_horizon_targets(y_test_base, horizon)
        valid_test = ~np.isnan(y_test_tree_2d).any(axis=1)
        X_test = X_test_base[valid_test]
        y_test_horizon = y_test_tree_2d[valid_test]  # 2D: (n_samples, horizon)

        for model_name in config.model_names:
            model_path = config.model_dir / f"{model_name.lower()}_{horizon}h.pkl"
            if not model_path.exists():
                print(f"  [SKIP] Model not found: {model_path}")
                continue

            try:
                model = load_model(
                    model_path, config, config.feature_cols if hasattr(config, 'feature_cols') else None,
                    horizon
                )
                model.config = model_config

                # Predict & align targets (single predict call, explicit logic)
                is_rnn = is_rnn_model(model.name)
                y_pred, y_true_aligned, n_horizons = align_targets(
                    model, X_test_base, X_test, y_test_base, y_test_horizon,
                    horizon, config.lookback, is_rnn
                )

                # Debug: check prediction stats
                print(f"    [DEBUG] {model.name} pred: mean={y_pred.mean():.1f}, std={y_pred.std():.1f}, shape={y_pred.shape}")
                print(f"    [DEBUG] True: mean={y_true_aligned.mean():.1f}, std={y_true_aligned.std():.1f}, shape={y_true_aligned.shape}")

                # Check for NaN
                if np.isnan(y_pred).any():
                    print(f"  [ERROR] {model.name} {horizon}h: Prediction contains NaN")
                    continue

                # Align lengths (should match, but defensive)
                if len(y_pred) != len(y_true_aligned):
                    min_len = min(len(y_pred), len(y_true_aligned))
                    y_pred = y_pred[:min_len]
                    y_true_aligned = y_true_aligned[:min_len]

                # Compute metrics (matches training Evaluator.calculate_metrics exactly)
                metrics = compute_metrics(
                    y_true_aligned, y_pred, n_horizons,
                    model.name, horizon, region
                )
                evaluator.add_result(metrics)
                print(f"  {model.name} {horizon}h - MAPE: {metrics['MAPE']:.2f}%, R2: {metrics['R2']:.2f}%")

            except Exception as e:
                print(f"  [ERROR] {model_name} {horizon}h: {e}")
                import traceback
                traceback.print_exc()
                continue


def plot_results(config, results_df, output_dir):
    """Generate validation plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MAPE heatmap
    pivot = results_df.pivot_table(index='Model', columns=['Region', 'Horizon'], values='MAPE')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('MAPE (%) Heatmap by Model, Region, and Horizon')
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_mape_heatmap.png', dpi=300)
    plt.close()

    # 2. Per-region bar charts
    for region in sorted(results_df['Region'].unique()):
        reg_df = results_df[results_df['Region'] == region]
        fig, axes = plt.subplots(1, len(config.horizons), figsize=(6*len(config.horizons), 5))
        if len(config.horizons) == 1:
            axes = [axes]

        for idx, h in enumerate(config.horizons):
            ax = axes[idx]
            h_df = results_df[(results_df['Region'] == region) & (results_df['Horizon'] == h)]
            if len(h_df) == 0:
                continue
            x = np.arange(len(h_df))
            width = 0.15
            ax.bar(x - 1.5*0.15, h_df['MAE'], 0.15, label='MAE')
            ax.bar(x - 0.5*0.15, h_df['RMSE'], 0.15, label='RMSE')
            ax.bar(x + 0.5*0.15, h_df['MAPE']*0.1, 0.15, label='MAPE*0.1')
            ax.bar(x + 1.5*0.15, h_df['R2'], 0.15, label='R2%')
            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title(f'{region} - {h}h')
            ax.set_xticks(x)
            ax.set_xticklabels(h_df['Model'], rotation=45)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'validation_{region}.png', dpi=300)
        plt.close()

    # 3. Summary bar chart (all regions)
    fig, axes = plt.subplots(len(config.horizons), 1, figsize=(10, 5*len(config.horizons)))
    if len(config.horizons) == 1:
        axes = [axes]

    for idx, h in enumerate(config.horizons):
        ax = axes[idx]
        h_df = results_df[results_df['Horizon'] == h]
        if len(h_df) == 0:
            continue
        pivot = h_df.pivot(index='Model', columns='Region', values='MAPE')
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(f'{h}h Horizon - MAPE by Region')
        ax.set_ylabel('MAPE (%)')
        ax.legend(title='Region')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_mape_by_region.png', dpi=300)
    plt.close()

    print(f"  [SAVE] Plots saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cross-Region Model Validation')
    parser.add_argument('--horizons', type=str, default='1,24,168', help='Horizons to validate')
    parser.add_argument('--models', type=str, default='xgboost,lightgbm,lstm,gru,bilstm,bigru,ann', help='Models to validate')
    parser.add_argument('--regions', type=str, default='NSW1,QLD1,SA1,TAS1,VIC1', help='Regions to validate')
    parser.add_argument('--data-file', type=str, default=f'aemo_features_{YEARS[0]}_{YEARS[-1]}.csv')
    parser.add_argument('--model-dir', type=str, default='./results/models', help='Directory with saved models')
    parser.add_argument('--output-dir', type=str, default='./results/validation', help='Output directory')
    parser.add_argument('--no-adapt-target', action='store_true', help='Disable region-specific target scaling adaptation')
    parser.add_argument('--adapt-target', action='store_true', help='Enable region-specific target scaling adaptation')
    parser.add_argument('--test-year', type=int, default=YEARS[-1], help='Test year (default: last year in YEARS)')
    parser.add_argument('--region', type=str, default=SELECT_REGION, help='Reference region for validation')
    parser.add_argument('--calibration-ratio', type=float, default=0.1, help='Ratio of test data to use for target scaler calibration')
    args = parser.parse_args()

    config = Config(
        horizons=[int(h) for h in args.horizons.split(',')],
        models=args.models.split(',') if args.models != 'all' else ['xgboost', 'lightgbm', 'lstm', 'gru', 'bilstm', 'bigru'],
        region=args.region,
        test_year=args.test_year
    )
    config.model_dir = Path(args.model_dir)
    config.output_dir = Path(args.output_dir)
    config.data_file = args.data_file
    config.adapt_target_scaler = args.adapt_target or (not args.no_adapt_target and config.adapt_target_scaler)
    config.calibration_ratio = args.calibration_ratio

    print("=" * 80)
    print("CROSS-REGION MODEL VALIDATION")
    print("=" * 80)
    print(f"Regions: {args.regions}")
    print(f"Horizons: {config.horizons}")
    print(f"Models: {config.model_names}")
    print(f"Adapt target scaler: {config.adapt_target_scaler}")
    print(f"Calibration ratio: {config.calibration_ratio}")
    print(f"Model dir: {config.model_dir}")
    print(f"Output dir: {config.output_dir}")
    print("=" * 80)

    # Data processor
    processor = DataProcessor(config)

    evaluator = Evaluator(config)
    regions = args.regions.split(',')

    for region in regions:
        test_df, feature_cols = processor.load_and_prepare(region)
        evaluate_region(config, region, test_df, feature_cols, evaluator)

    # Save results
    results_df = evaluator.get_results_df()
    evaluator.save_report(config.output_dir)
    print(f"\n[SAVE] Results saved to {config.output_dir}")

    # Generate plots
    plot_results(config, evaluator.get_results_df(), config.output_dir)

    print("\n[DONE] Cross-region validation completed!")
    print(f"[INFO] Results saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
