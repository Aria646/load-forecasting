"""
Function: Train and compare multiple short-term load forecasting models
Project: Energy Data Analysis - Short Term Load Forecasting (STLF)

Models:
1. XGBoost - Tree model baseline
2. LightGBM - Tree model baseline
3. LSTM - Long Short-Term Memory
4. GRU - Gated Recurrent Unit
5. BiLSTM - Bidirectional LSTM
6. BiGRU - Bidirectional GRU
7. ANN - Artificial Neural Network (baseline)

Metrics: MAE, MAPE, RMSE, R2
Prediction Horizons: 1h, 24h, 168h (1 week)
"""

import argparse
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

warnings.filterwarnings('ignore')


# ============================================================
# Shared Utilities
# ============================================================

def create_horizon_targets(y_base: np.ndarray, horizon: int) -> np.ndarray:
    """
    Create multi-horizon targets from 1D base array.
    
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


# ============================================================
# Model Architectures (module-level for pickling)
# ============================================================

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

        # Hybrid: RNN output + explicit lag_168
        self.fc1 = nn.Linear(rnn_hidden + 1, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, features]
        lag_168_feat = x[:, 0, self.lag_168_idx:self.lag_168_idx+1] if self.lag_168_idx >= 0 else torch.zeros(x.size(0), 1, device=x.device)
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        combined = torch.cat([last_hidden, lag_168_feat], dim=1)
        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ANN(nn.Module):
    """Feedforward Neural Network"""

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
# Configuration & CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='STLF Model Training')
    parser.add_argument('--horizons', type=str, default='1,24,168',
                        help='Prediction horizons (hours), comma-separated. e.g., 1,24,168')
    parser.add_argument('--models', type=str, default='all',
                        help='Models to train, comma-separated. e.g., xgboost,lightgbm,lstm,gru,bilstm,bigru,ann')
    parser.add_argument('--data-file', type=str, default=f'aemo_features_{YEARS[0]}_{YEARS[-1]}.csv',
                        help='Input data file path')
    parser.add_argument('--region', type=str, default=f'{SELECT_REGION}',
                        help='Region to model')
    parser.add_argument('--lookback', type=int, default=168,
                        help='Lookback window (hours)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Max epochs for neural networks')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for neural networks')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size for RNN')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for neural networks')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results and models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


class Config:
    """Configuration container"""
    def __init__(self, args):
        self.horizons = [int(h) for h in args.horizons.split(',')]
        self.model_names = args.models.split(',') if args.models != 'all' else [
            'xgboost', 'lightgbm', 'lstm', 'gru', 'bilstm', 'bigru', 'ann'
        ]
        self.data_file = args.data_file
        self.region = args.region
        self.lookback = args.lookback
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.lr = args.lr
        self.output_dir = Path(args.output_dir)
        self.seed = args.seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Base Classes
# ============================================================

class ModelBase(ABC):
    """Abstract base class for all models"""

    def __init__(self, name: str, horizon: int, config: Config):
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


class DataProcessor:
    """Handle data loading, cleaning, splitting, and sequencing"""

    def __init__(self, config: Config):
        self.config = config
        self.feature_cols = None
        self.region_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_and_prepare(self):
        """Load data, clean, split by year, prepare features"""
        print(f"\n[DATA] 1. Loading data from {self.config.data_file}...")

        if not Path(self.config.data_file).exists():
            raise FileNotFoundError(f"Data file not found: {self.config.data_file}")

        df = pd.read_csv(self.config.data_file)
        df['HOUR'] = pd.to_datetime(df['HOUR'])
        print(f"  [OK] Loaded: {len(df):,} records")
        print(f"  [INFO] Date range: {df['HOUR'].min()} to {df['HOUR'].max()}")
        print(f"  [INFO] Regions: {df['REGIONID'].unique().tolist()}")

        # Select region
        region_df = df[df['REGIONID'] == self.config.region].copy()
        print(f"  [INFO] Selected region: {self.config.region} ({len(region_df):,} records)")

        # Drop columns with severe missing values
        drop_cols = ['VISIBILITY', 'PRECIPITATION', 'WIND_SPEED', 'REGIONID']
        region_df = region_df.drop(columns=[c for c in drop_cols if c in region_df.columns])

        # Fill minor missing weather data
        weather_cols = ['TEMPERATURE', 'DEW_POINT', 'HUMIDITY', 'PRESSURE']
        for c in weather_cols:
            if c in region_df.columns:
                region_df[c] = region_df[c].fillna(region_df[c].median())

        # Fill remaining with 0
        region_df = region_df.fillna(0)

        # Sort by time
        region_df = region_df.sort_values('HOUR').reset_index(drop=True)
        self.region_df = region_df

        # Split by year: {YEARS[0]}-{YEARS[-2]} train, {YEARS[-1]} test
        train_df = region_df[region_df['HOUR'].dt.year <= YEARS[-2]].copy()
        test_df = region_df[region_df['HOUR'].dt.year == YEARS[-1]].copy()

        # Validation split for tree models (last 10% of train)
        val_split = int(len(train_df) * 0.9)
        val_df = train_df.iloc[val_split:].copy()
        train_df = train_df.iloc[:val_split].copy()

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        print(f"  [INFO] Train ({YEARS[0]}-{YEARS[-2]}): {len(train_df):,} records")
        print(f"  [INFO] Val (last 10%): {len(val_df):,} records")
        print(f"  [INFO] Test ({YEARS[-1]}): {len(test_df):,} records")

        # Define feature columns
        exclude_cols = ['HOUR', 'TOTALDEMAND']
        self.feature_cols = [c for c in region_df.columns if c not in exclude_cols]
        print(f"  [INFO] Feature count: {len(self.feature_cols)}")

        return self.train_df, self.val_df, self.test_df, self.feature_cols

    @staticmethod
    def create_sequences(X, y, lookback, horizon):
        """Create sliding window sequences for time series
        X: (n_samples, n_features)
        y: (n_samples,) or (n_samples, horizon)
        Returns: X_seq (n_seq, lookback, n_features), y_seq (n_seq, horizon)
        """
        X_seq, y_seq = [], []
        for i in range(lookback, len(X) - horizon + 1):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i:i+horizon])
        return np.array(X_seq), np.array(y_seq)


class Evaluator:
    """Model evaluation and visualization"""

    def __init__(self, config: Config):
        self.config = config
        self.results = []

    @staticmethod
    def calculate_metrics(y_true, y_pred, name, horizon, n_horizons=None):
        """
        Calculate metrics per horizon step and average.
        
        Args:
            y_true: Ground truth (n_samples, n_horizons) or (n_samples,)
            y_pred: Predictions (n_samples, n_horizons) or (n_samples,)
            name: Model name
            horizon: Prediction horizon (for reporting)
            n_horizons: Number of horizon steps to evaluate. 
                       If None, inferred from y_true.shape[1].
                       For RNN models, pass 1 to evaluate only first step.
        """
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Ensure 2D for per-horizon calculation
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Explicit n_horizons: allows evaluating subset of prediction horizons
        if n_horizons is None:
            n_horizons = y_true.shape[1]
        
        mape_per_h = []
        mae_per_h = []
        rmse_per_h = []
        r2_per_h = []

        for h in range(n_horizons):
            yt = y_true[:, h]
            # For RNN: y_pred may have more columns than n_horizons, take available
            yp = y_pred[:, h] if y_pred.shape[1] > h else y_pred[:, 0]

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
            'Model': name,
            'Horizon': horizon,
            'MAE': np.mean(mae_per_h),
            'MAE_per_h': mae_per_h,
            'MAPE': np.mean(mape_per_h),
            'MAPE_per_h': mape_per_h,
            'RMSE': np.mean(rmse_per_h),
            'RMSE_per_h': rmse_per_h,
            'R2': np.mean(r2_per_h) * 100,
            'R2_per_h': [r * 100 for r in r2_per_h],
        }

    def add_result(self, metrics):
        self.results.append(metrics)

    def get_results_df(self):
        df = pd.DataFrame(self.results)
        return df.round(4)

    def print_summary(self, horizons: str):
        df = self.get_results_df()
        print("\n" + "=" * 80)
        print(f"MODEL PERFORMANCE COMPARISON ({horizons})")
        print("=" * 80)
        summary = df[['Model', 'Horizon', 'MAE', 'MAPE', 'RMSE', 'R2']].copy()
        print(summary.to_string(index=False))
        print("=" * 80)

    def save_report(self, output_dir: Path):
        df = self.get_results_df()
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        # df.to_csv(output_dir / f'metrics_summary_{horizon}h.csv', index=False)
        summary_cols = ['Model', 'Horizon', 'MAE', 'MAPE', 'RMSE', 'R2']
        summary_df = df[summary_cols].copy()
        summary_df.to_csv(output_dir / 'metrics_summary.csv', index=False)

        # Text report
        with open(output_dir / f'model_report.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("STLF Model Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Region: {self.config.region}\n")
            f.write(f"Horizons: {self.config.horizons}\n")
            f.write(f"Lookback: {self.config.lookback}h\n")
            f.write(f"Models: {', '.join(self.config.model_names)}\n\n")
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")

            # Best per horizon
            f.write("BEST MODEL PER HORIZON:\n")
            f.write("-" * 80 + "\n")
            for h in self.config.horizons:
                h_df = df[df['Horizon'] == h]
                if len(h_df) > 0:
                    best = h_df.loc[h_df['MAPE'].idxmin()]
                    f.write(f"  {h}h: {best['Model']} (MAPE={best['MAPE']:.2f}%)\n")


# ============================================================
# Model Implementations
# ============================================================

class XGBoostModel(ModelBase):
    """XGBoost with multi-horizon support via independent models per horizon step"""

    def fit(self, X_train, y_train, X_val, y_val, feature_cols):
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        # Ensure 2D for multi-horizon: (n_samples, horizon)
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_train_scaled = self.target_scaler.fit_transform(y_train_2d)
        y_val_scaled = self.target_scaler.transform(y_val_2d)

        # Train independent model for each horizon step
        self.models = []
        for h in range(self.horizon):
            model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=30,
                random_state=self.config.seed,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(
                X_train_scaled, y_train_scaled[:, h],
                eval_set=[(X_val_scaled, y_val_scaled[:, h])],
                verbose=False
            )
            self.models.append(model)

        self.model = self.models  # Store list of models
        return self

    def predict(self, X):
        X_scaled = self.feature_scaler.transform(X)
        preds = np.column_stack([m.predict(X_scaled) for m in self.model])
        preds = self.target_scaler.inverse_transform(preds)
        return preds


class LightGBMModel(ModelBase):
    """LightGBM with multi-horizon support"""

    def fit(self, X_train, y_train, X_val, y_val, feature_cols):
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_train_scaled = self.target_scaler.fit_transform(y_train_2d)
        y_val_scaled = self.target_scaler.transform(y_val_2d)

        self.models = []
        for h in range(self.horizon):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.seed,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(
                X_train_scaled, y_train_scaled[:, h],
                eval_set=[(X_val_scaled, y_val_scaled[:, h])],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            self.models.append(model)

        self.model = self.models
        return self

    def predict(self, X):
        X_scaled = self.feature_scaler.transform(X)
        preds = np.column_stack([m.predict(X_scaled) for m in self.model])
        preds = self.target_scaler.inverse_transform(preds)
        return preds


class ANNModel(ModelBase):
    """Feedforward Neural Network"""

    def fit(self, X_train, y_train, X_val, y_val, feature_cols):
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)

        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        # Ensure 2D for multi-horizon: (n_samples, horizon)
        y_train_2d = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        y_val_2d = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
        y_train_scaled = self.target_scaler.fit_transform(y_train_2d)
        y_val_scaled = self.target_scaler.transform(y_val_2d)

        input_dim = X_train_scaled.shape[1]
        output_dim = y_train_scaled.shape[1]

        self.model = ANN(input_dim, output_dim).to(self.config.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5)

        X_train_t = torch.FloatTensor(X_train_scaled).to(self.config.device)
        y_train_t = torch.FloatTensor(y_train_scaled).to(self.config.device)
        X_val_t = torch.FloatTensor(X_val_scaled).to(self.config.device)
        y_val_t = torch.FloatTensor(y_val_scaled).to(self.config.device)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                                batch_size=self.config.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = self.model(bx)
                    val_loss += criterion(pred, by).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.config.epochs}, Train: {train_loss/len(train_loader):.6f}, Val: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(best_state)
                break

        return self

    def predict(self, X):
        X_scaled = self.feature_scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_t).cpu().numpy()
        pred = self.target_scaler.inverse_transform(pred)
        return pred


class LSTMModel(ModelBase):
    """LSTM/GRU/BiLSTM/BiGRU with hybrid architecture (RNN output + explicit lag_168)"""

    def __init__(self, name: str, horizon: int, config: Config, variant: str = 'lstm'):
        super().__init__(name, horizon, config)
        self.variant = variant  # 'lstm', 'gru', 'bilstm', 'bigru'

    def fit(self, X_train, y_train, X_val, y_val, feature_cols):
        # If val data not provided, use full train data and split sequences internally
        if X_val is None or y_val is None:
            X_full = X_train
            y_full = y_train
            # No separate val set - will split sequences 90/10 later
            X_train_only = X_train
            y_train_only = y_train
        else:
            X_full = np.vstack([X_train, X_val])
            y_full = np.concatenate([y_train, y_val])
            X_train_only = X_train
            y_train_only = y_train

        # Feature scaling on train only
        self.feature_scaler = StandardScaler()
        X_full_scaled = self.feature_scaler.fit_transform(X_full)

        # Target scaling on train only - NO data leakage
        # Fit scaler ONLY on training targets (not validation)
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_train_2d = y_train_only.reshape(-1, 1)
        self.target_scaler.fit(y_train_2d)
        # Transform full data using train-fitted scaler
        y_full_scaled = self.target_scaler.transform(y_full.reshape(-1, 1)).flatten()

        # Create sequences from full scaled data
        processor = DataProcessor(self.config)
        X_full_seq, y_full_seq = processor.create_sequences(
            X_full_scaled, y_full_scaled, self.config.lookback, self.horizon)

        # Split sequences 90/10 for validation
        val_split = int(len(X_full_seq) * 0.9)
        X_train_seq = X_full_seq[:val_split]
        y_train_seq = y_full_seq[:val_split]
        X_val_seq = X_full_seq[val_split:]
        y_val_seq = y_full_seq[val_split:]

        print(f"      Sequences: Train {len(X_train_seq)}, Val {len(X_val_seq)}, Horizon {self.horizon}, Features {len(feature_cols)}")

        # PyTorch tensors
        X_train_t = torch.FloatTensor(X_train_seq)
        y_train_t = torch.FloatTensor(y_train_seq)
        X_val_t = torch.FloatTensor(X_val_seq)
        y_val_t = torch.FloatTensor(y_val_seq)

        # Ensure target tensors are 2D: (n_samples, horizon)
        if y_train_t.ndim == 1:
            y_train_t = y_train_t.reshape(-1, 1)
        if y_val_t.ndim == 1:
            y_val_t = y_val_t.reshape(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                  batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                                batch_size=self.config.batch_size, shuffle=False)

        # Model definition
        input_size = len(feature_cols)
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        output_size = self.horizon

        # Find LOAD_LAG_168 index
        lag_168_idx = feature_cols.index('LOAD_LAG_168') if 'LOAD_LAG_168' in feature_cols else -1

        self.model = HybridRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            variant=self.variant,
            lag_168_idx=lag_168_idx
        ).to(self.config.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(self.config.device), by.to(self.config.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.config.device), by.to(self.config.device)
                    pred = self.model(bx)
                    val_loss += criterion(pred, by).item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{self.config.epochs}, Train: {train_loss/len(train_loader):.6f}, Val: {val_loss:.6f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}, best_val_loss={best_val_loss:.6f}")
                if best_state:
                    self.model.load_state_dict(best_state)
                break

        return self

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


# ============================================================
# Model Factory
# ============================================================

def create_model(name: str, horizon: int, config: Config):
    """Factory function to create model instances"""
    name_lower = name.lower()
    if name_lower == 'xgboost':
        return XGBoostModel('XGBoost', horizon, config)
    elif name_lower == 'lightgbm':
        return LightGBMModel('LightGBM', horizon, config)
    elif name_lower == 'ann':
        return ANNModel('ANN', horizon, config)
    elif name_lower in ['lstm', 'gru', 'bilstm', 'bigru']:
        return LSTMModel(name.upper(), horizon, config, variant=name_lower)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================
# Main Training Pipeline
# ============================================================

def run_experiment(config: Config):
    """Main experiment runner"""
    print("=" * 80)
    print("SHORT-TERM LOAD FORECASTING (STLF) - MODEL TRAINING")
    print("=" * 80)
    print(f"Region: {config.region}")
    print(f"Horizons: {config.horizons}h")
    print(f"Models: {', '.join(config.model_names)}")
    print(f"Lookback: {config.lookback}h")
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output: {config.output_dir}")
    print("=" * 80)

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data preparation
    processor = DataProcessor(config)
    # Get full training data (before validation split) for LSTM models
    full_train_df, val_df, test_df, feature_cols = processor.load_and_prepare()

    # For tree models, split train into train/val (90/10)
    # For LSTM models, use full train data and let them do sequence-level validation split
    train_df = full_train_df  # Use full data for LSTM
    # For tree models, we'll split here
    val_split = int(len(full_train_df) * 0.9)
    X_train_tree = full_train_df.iloc[:val_split][feature_cols].fillna(0).values
    y_train_tree = full_train_df.iloc[:val_split]['TOTALDEMAND'].values
    X_val_tree = full_train_df.iloc[val_split:][feature_cols].fillna(0).values
    y_val_tree = full_train_df.iloc[val_split:]['TOTALDEMAND'].values

    # Test data (same for all models)
    X_test_base = test_df[feature_cols].fillna(0).values
    y_test_base = test_df['TOTALDEMAND'].values

    # Full training data for LSTM (no DataFrame-level split)
    X_train_lstm = full_train_df[feature_cols].fillna(0).values
    y_train_lstm = full_train_df['TOTALDEMAND'].values

    # Evaluator
    evaluator = Evaluator(config)

    # Create model output directory
    models_dir = config.output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    horizons = ''

    # Training loop: for each horizon, for each model
    for horizon in config.horizons:
        print(f"\n{'='*80}")
        print(f"HORIZON: {horizon}h")
        print(f"{'='*80}")

        if len(horizons) > 0:
            horizons += ', '
        horizons += str(horizon) + 'h'

        # Tree models: use tree-split data (1D -> 2D)
        y_train_tree_2d = create_horizon_targets(y_train_tree, horizon)
        y_val_tree_2d = create_horizon_targets(y_val_tree, horizon)

        # Test: use test base data
        y_test_tree_2d = create_horizon_targets(y_test_base, horizon)

        # Remove rows with NaN (last horizon-1 samples)
        valid_train = ~np.isnan(y_train_tree_2d).any(axis=1)
        valid_val = ~np.isnan(y_val_tree_2d).any(axis=1)
        valid_test = ~np.isnan(y_test_tree_2d).any(axis=1)

        X_train = X_train_tree[valid_train]
        y_train = y_train_tree_2d[valid_train]
        X_val = X_val_tree[valid_val]
        y_val = y_val_tree_2d[valid_val]
        X_test = X_test_base[valid_test]
        y_test = y_test_tree_2d[valid_test]

        print(f"  [INFO] Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        for model_name in config.model_names:
            print(f"\n[MODEL] Training {model_name.upper()} (horizon={horizon}h)...")

            try:
                model = create_model(model_name, horizon, config)

                # LSTM/GRU models need 1D raw targets for sequence creation
                # Tree models and ANN use pre-prepared 2D targets
                if isinstance(model, LSTMModel):
                    # Use full training data for LSTM (with sequence-level validation split)
                    model.fit(X_train_lstm, y_train_lstm, None, None, feature_cols)
                else:
                    # Tree models and ANN use pre-prepared 2D targets (already filtered)
                    model.fit(X_train, y_train, X_val, y_val, feature_cols)

                # Predict on test set
                if isinstance(model, LSTMModel):
                    y_pred = model.predict(X_test_base)
                    # LSTM predictions align with test data starting from lookback index
                    y_test_aligned = y_test_base[config.lookback:config.lookback + len(y_pred)]
                    # RNN: explicitly evaluate only first horizon step (matches training behavior)
                    n_horizons_eval = 1
                else:
                    y_pred = model.predict(X_test)
                    y_test_aligned = y_test
                    # Tree/ANN: evaluate all horizon steps
                    n_horizons_eval = horizon

                # Evaluate
                metrics = Evaluator.calculate_metrics(y_test_aligned, y_pred, model.name, horizon, n_horizons=n_horizons_eval)
                evaluator.add_result(metrics)

                print(f"  [RESULT] {model.name} {horizon}h - MAPE: {metrics['MAPE']:.2f}%, R2: {metrics['R2']:.2f}%")

                # Save model
                model_path = models_dir / f"{model.name.lower()}_{horizon}h.pkl"
                model.save(model_path)

            except Exception as e:
                print(f"  [ERROR] Failed to train {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Final summary and save for this horizon
        evaluator.print_summary(horizons)
        evaluator.save_report(config.output_dir)

        # Save visualizations
        plot_results(config, evaluator.get_results_df(), test_df, config.output_dir)

    print(f"\n[DONE] All experiments completed!")
    print(f"[INFO] Results saved to: {config.output_dir}")
    print(f"[INFO] Models saved to: {models_dir}")


def plot_results(config: Config, results_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance comparison bar chart (grouped by horizon)
    fig, axes = plt.subplots(1, len(config.horizons), figsize=(6*len(config.horizons), 5))
    if len(config.horizons) == 1:
        axes = [axes]

    for idx, h in enumerate(config.horizons):
        ax = axes[idx]
        h_df = results_df[results_df['Horizon'] == h]
        if len(h_df) == 0:
            continue

        x = np.arange(len(h_df))
        width = 0.15

        ax.bar(x - 1.5*width, h_df['MAE'], width, label='MAE')
        ax.bar(x - 0.5*width, h_df['RMSE'], width, label='RMSE')
        ax.bar(x + 0.5*width, h_df['MAPE'] * 0.1, width, label='MAPE*0.1')
        ax.bar(x + 1.5*width, h_df['R2'], width, label='R2%')

        ax.set_xlabel('Model')
        ax.set_ylabel('Metric Value')
        ax.set_title(f'{h}h Horizon')
        ax.set_xticks(x)
        ax.set_xticklabels(h_df['Model'], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300)
    plt.close()

    # 2. Per-horizon MAPE heatmap
    pivot = results_df.pivot(index='Model', columns='Horizon', values='MAPE')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('MAPE (%) Heatmap by Model and Horizon')
    plt.tight_layout()
    plt.savefig(output_dir / 'mape_heatmap.png', dpi=300)
    plt.close()

    print(f"  [SAVE] Plots saved to {output_dir}")


# ============================================================
# Entry Point
# ============================================================

def main():
    args = parse_args()
    config = Config(args)
    run_experiment(config)


if __name__ == '__main__':
    main()
