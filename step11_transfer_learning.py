"""
Function: Transfer learning for cross-state load forecasting
Project: Energy Data Analysis - Short Term Load Forecasting (STLF)

Loads NSW-pretrained models and adapts them to target states (QLD1, SA1, TAS1, VIC1)
using four strategies:
  1. scaler   - refit target scaler only
  2. partial  - freeze early layers, retrain output layers
  3. full     - full fine-tuning with small learning rate
  4. fewshot  - systematic study of target data size

Metrics: MAE, MAPE, RMSE, R2
Outputs:
  - ./results/transfer/         (CSV + plots)
  - ./results/transfer_models/  (fine-tuned models, separate from ./results/models/)
"""

import argparse
import pickle
import warnings
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
# Model Architectures (must match training exactly for pickle)
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

        self.fc1 = nn.Linear(rnn_hidden + 1, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lag_168_feat = x[:, 0, self.lag_168_idx:self.lag_168_idx+1] if self.lag_168_idx >= 0 \
            else torch.zeros(x.size(0), 1, device=x.device)
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
# Utilities
# ============================================================

def is_rnn_model(name: str) -> bool:
    return any(v in name.lower() for v in ('lstm', 'gru'))


def create_sequences(X, y, lookback, horizon):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X) - horizon + 1):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i:i+horizon])
    return np.array(X_seq), np.array(y_seq)


def create_horizon_targets(y_base, horizon):
    """
    Same as training's create_horizon_targets.
    Returns (n_samples, horizon) array with NaN padding at the end.
    Column order: [t+h, t+h-1, ..., t+1] (reverse chronological).
    """
    n = len(y_base)
    targets = np.zeros((n, horizon))
    for i in range(horizon):
        targets[:-horizon+i, i] = y_base[horizon-i:] if horizon-i > 0 else y_base[:]
        if horizon-i > 0:
            targets[-(horizon-i):, i] = np.nan
    return targets


def evaluate_predictions(model_name, horizon, pred, target_test_df, config):
    """
    Unified evaluation: handles alignment for RNN vs. tree/ANN.
    Returns (y_true, y_pred, n_horizons) ready for compute_metrics.
    """
    if is_rnn_model(model_name):
        # RNN: predictions start at index `lookback` in test set
        y_true = target_test_df['TOTALDEMAND'].values[
            config['lookback']:config['lookback']+len(pred)]
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        n_h = 1
    else:
        # Tree/ANN: predictions are (n_samples, horizon) over ALL test rows
        y_base = target_test_df['TOTALDEMAND'].values
        y_2d = create_horizon_targets(y_base, horizon)
        valid = ~np.isnan(y_2d).any(axis=1)
        y_true = y_2d[valid]
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        pred = pred[:len(y_true)]
        n_h = horizon

    return y_true, pred, n_h


def compute_metrics(y_true, y_pred, n_horizons, model_name, horizon, region, strategy):
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
        yp = y_pred[:, h] if y_pred.shape[1] > h else y_pred[:, 0]

        mae = mean_absolute_error(yt, yp)
        yt_safe = np.where(yt == 0, 1e-6, yt)
        mape = np.mean(np.abs((yt - yp) / yt_safe)) * 100
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp) * 100

        mae_per_h.append(mae)
        mape_per_h.append(mape)
        rmse_per_h.append(rmse)
        r2_per_h.append(r2)

    return {
        'Region': region,
        'Model': model_name,
        'Horizon': horizon,
        'Strategy': strategy,
        'MAE': round(np.mean(mae_per_h), 2),
        'MAPE': round(np.mean(mape_per_h), 2),
        'RMSE': round(np.mean(rmse_per_h), 2),
        'R2': round(np.mean(r2_per_h), 2),
    }


def save_transfer_model(model_obj, feature_scaler, target_scaler, meta, save_path):
    """
    Save a fine-tuned transfer model to disk.
    Works for both tree models (list of per-horizon estimators or Boosters) and PyTorch models.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_obj, nn.Module):
        model_obj = model_obj.cpu()

    payload = {
        'model': model_obj,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'horizon': meta['horizon'],
        'name': meta['name'],
        'source_region': meta.get('source_region', 'NSW1'),
        'target_region': meta['target_region'],
        'strategy': meta['strategy'],
        'model_type': meta['model_type'],
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)


# ============================================================
# Data Loading
# ============================================================

class TargetDataLoader:
    """Load and prepare target-state data for transfer learning"""

    def __init__(self, data_file, target_region, test_year=2019):
        self.data_file = data_file
        self.target_region = target_region
        self.test_year = test_year
        self.lookback = 168

    def load(self):
        df = pd.read_csv(self.data_file)
        df['HOUR'] = pd.to_datetime(df['HOUR'])

        region_df = df[df['REGIONID'] == self.target_region].copy()

        drop_cols = ['VISIBILITY', 'PRECIPITATION', 'WIND_SPEED', 'REGIONID']
        region_df = region_df.drop(columns=[c for c in drop_cols if c in region_df.columns])

        weather_cols = ['TEMPERATURE', 'DEW_POINT', 'HUMIDITY', 'PRESSURE']
        for c in weather_cols:
            if c in region_df.columns:
                region_df[c] = region_df[c].fillna(region_df[c].median())

        region_df = region_df.fillna(0).sort_values('HOUR').reset_index(drop=True)

        exclude_cols = ['HOUR', 'TOTALDEMAND']
        feature_cols = [c for c in region_df.columns if c not in exclude_cols]

        train_df = region_df[region_df['HOUR'].dt.year < self.test_year].copy()
        test_df = region_df[region_df['HOUR'].dt.year == self.test_year].copy()

        return train_df, test_df, feature_cols


# ============================================================
# Strategy 1: Scaler Adaptation
# ============================================================

def strategy_scaler(model_name, horizon, nsw_model_dict, target_train_df,
                    target_test_df, feature_cols, config):
    """
    Refit target_scaler on target-state training data; keep feature_scaler from source.
    Returns (predictions, target_scaler, model_to_save).
    """
    name_lower = model_name.lower()

    if 'xgboost' in name_lower or 'lightgbm' in name_lower:
        models = nsw_model_dict['model']
        feature_scaler = nsw_model_dict['feature_scaler']

        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_target = target_train_df['TOTALDEMAND'].values.reshape(-1, 1)
        target_scaler.fit(y_target)

        X_test = target_test_df[feature_cols].fillna(0).values
        X_scaled = feature_scaler.transform(X_test)
        preds = np.column_stack([m.predict(X_scaled) for m in models])
        preds = target_scaler.inverse_transform(preds)

        return preds, target_scaler, models

    elif 'ann' in name_lower:
        model = nsw_model_dict['model']
        feature_scaler = nsw_model_dict['feature_scaler']

        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_target = target_train_df['TOTALDEMAND'].values.reshape(-1, 1)
        target_scaler.fit(y_target)

        device = config['device']
        X_test = target_test_df[feature_cols].fillna(0).values
        X_scaled = feature_scaler.transform(X_test)
        X_t = torch.FloatTensor(X_scaled).to(device)

        model.eval()
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        pred = target_scaler.inverse_transform(pred)
        return pred, target_scaler, model

    else:
        model = nsw_model_dict['model']
        feature_scaler = nsw_model_dict['feature_scaler']

        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_target = target_train_df['TOTALDEMAND'].values.reshape(-1, 1)
        target_scaler.fit(y_target)

        device = config['device']
        X_test = target_test_df[feature_cols].fillna(0).values
        X_scaled = feature_scaler.transform(X_test)

        X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)),
                                    config['lookback'], horizon)
        X_t = torch.FloatTensor(X_seq).to(device)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), config['batch_size']):
                batch = X_t[i:i+config['batch_size']]
                preds.append(model(batch).cpu().numpy())
        y_pred_scaled = np.vstack(preds)
        y_pred_scaled = np.clip(y_pred_scaled, -1.0, 1.0)
        orig_shape = y_pred_scaled.shape
        y_pred_flat = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_flat.reshape(orig_shape), target_scaler, model


# ============================================================
# Strategy 2: Partial Fine-tuning
# ============================================================

def strategy_partial(model_name, horizon, nsw_model_dict, target_train_df,
                     target_test_df, feature_cols, config):
    """
    Freeze early layers, retrain output layers on target data.
    Returns (predictions, target_scaler, fine_tuned_model).
    """
    name_lower = model_name.lower()
    device = config['device']

    feature_scaler = nsw_model_dict['feature_scaler']
    X_target = target_train_df[feature_cols].fillna(0).values
    X_target_scaled = feature_scaler.transform(X_target)

    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_target = target_train_df['TOTALDEMAND'].values.reshape(-1, 1)
    target_scaler.fit(y_target)

    if 'xgboost' in name_lower:
        # === XGBoost: use Booster API (xgb.train) for reliable continued training ===
        models = nsw_model_dict['model']
        new_boosters = []
        for h, base_model in enumerate(models):
            y_2d = np.full((len(y_target), horizon), np.nan)
            for i in range(horizon):
                if horizon - i > 0:
                    y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
            valid = ~np.isnan(y_2d[:, h])
            X_h = X_target_scaled[valid]
            y_h = target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()

            base_booster = base_model.get_booster()
            dtrain = xgb.DMatrix(X_h, label=y_h)
            params = {
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'seed': config['seed'],
            }
            new_booster = xgb.train(
                params, dtrain,
                num_boost_round=100,
                xgb_model=base_booster,
                verbose_eval=False
            )
            new_boosters.append(new_booster)

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)
        preds = np.column_stack([b.predict(dtest) for b in new_boosters])
        preds = target_scaler.inverse_transform(preds)
        return preds, target_scaler, new_boosters

    elif 'lightgbm' in name_lower:
        models = nsw_model_dict['model']
        new_models = []
        for h, base_model in enumerate(models):
            y_2d = np.full((len(y_target), horizon), np.nan)
            for i in range(horizon):
                if horizon - i > 0:
                    y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
            valid = ~np.isnan(y_2d[:, h])
            X_h = X_target_scaled[valid]
            y_h = target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()

            base_rounds = base_model.booster_.num_trees()
            new_model = lgb.LGBMRegressor(
                n_estimators=base_rounds + 100,
                learning_rate=0.05, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8,
                random_state=config['seed'], n_jobs=-1, verbose=-1
            )
            new_model.fit(X_h, y_h, init_model=base_model)
            new_models.append(new_model)

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        preds = np.column_stack([m.predict(X_test_scaled) for m in new_models])
        preds = target_scaler.inverse_transform(preds)
        return preds, target_scaler, new_models

    elif 'ann' in name_lower:
        model = nsw_model_dict['model']
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = False

        y_2d = np.full((len(y_target), horizon), np.nan)
        for i in range(horizon):
            if horizon - i > 0:
                y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
        valid = ~np.isnan(y_2d).any(axis=1)
        X_h = X_target_scaled[valid]
        # Transform each horizon column separately (scaler fitted on 1D)
        y_h = np.column_stack([
            target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()
            for h in range(horizon)
        ])

        model = model.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['finetune_lr']
        )
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_h).to(device)
        y_t = torch.FloatTensor(y_h).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=config['batch_size'], shuffle=True)

        model.train()
        for epoch in range(config['finetune_epochs']):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()
        pred = target_scaler.inverse_transform(pred)
        return pred, target_scaler, model

    else:
        # RNN partial fine-tuning
        model = nsw_model_dict['model']
        for param in model.rnn.parameters():
            param.requires_grad = False

        y_target_flat = target_train_df['TOTALDEMAND'].values
        y_target_scaled = target_scaler.transform(y_target_flat.reshape(-1, 1)).flatten()

        X_seq, y_seq = create_sequences(X_target_scaled, y_target_scaled,
                                        config['lookback'], horizon)

        model = model.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['finetune_lr']
        )
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_seq).to(device)
        y_t = torch.FloatTensor(y_seq).to(device)
        if y_t.ndim == 1:
            y_t = y_t.reshape(-1, 1)

        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=config['batch_size'], shuffle=True)

        model.train()
        for epoch in range(config['finetune_epochs']):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        X_seq_test, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)),
                                         config['lookback'], horizon)
        X_test_t = torch.FloatTensor(X_seq_test).to(device)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test_t), config['batch_size']):
                batch = X_test_t[i:i+config['batch_size']]
                preds.append(model(batch).cpu().numpy())
        y_pred_scaled = np.vstack(preds)
        y_pred_scaled = np.clip(y_pred_scaled, -1.0, 1.0)
        orig_shape = y_pred_scaled.shape
        y_pred_flat = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_flat.reshape(orig_shape), target_scaler, model


# ============================================================
# Strategy 3: Full Fine-tuning
# ============================================================

def strategy_full(model_name, horizon, nsw_model_dict, target_train_df,
                  target_test_df, feature_cols, config):
    """
    Full fine-tuning: all parameters trainable, small learning rate.
    Returns (predictions, target_scaler, fine_tuned_model).
    """
    name_lower = model_name.lower()
    device = config['device']

    feature_scaler = nsw_model_dict['feature_scaler']
    X_target = target_train_df[feature_cols].fillna(0).values
    X_target_scaled = feature_scaler.transform(X_target)

    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_target = target_train_df['TOTALDEMAND'].values.reshape(-1, 1)
    target_scaler.fit(y_target)

    if 'xgboost' in name_lower:
        # === XGBoost: use Booster API (xgb.train) for reliable continued training ===
        models = nsw_model_dict['model']
        new_boosters = []
        for h, base_model in enumerate(models):
            y_2d = np.full((len(y_target), horizon), np.nan)
            for i in range(horizon):
                if horizon - i > 0:
                    y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
            valid = ~np.isnan(y_2d[:, h])
            X_h = X_target_scaled[valid]
            y_h = target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()

            base_booster = base_model.get_booster()
            dtrain = xgb.DMatrix(X_h, label=y_h)
            params = {
                'learning_rate': config['finetune_lr_tree'],
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'reg:squarederror',
                'seed': config['seed'],
            }
            new_booster = xgb.train(
                params, dtrain,
                num_boost_round=200,
                xgb_model=base_booster,
                verbose_eval=False
            )
            new_boosters.append(new_booster)

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        dtest = xgb.DMatrix(X_test_scaled)
        preds = np.column_stack([b.predict(dtest) for b in new_boosters])
        preds = target_scaler.inverse_transform(preds)
        return preds, target_scaler, new_boosters

    elif 'lightgbm' in name_lower:
        models = nsw_model_dict['model']
        new_models = []
        for h, base_model in enumerate(models):
            y_2d = np.full((len(y_target), horizon), np.nan)
            for i in range(horizon):
                if horizon - i > 0:
                    y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
            valid = ~np.isnan(y_2d[:, h])
            X_h = X_target_scaled[valid]
            y_h = target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()

            base_rounds = base_model.booster_.num_trees()
            new_model = lgb.LGBMRegressor(
                n_estimators=base_rounds + 200,
                learning_rate=config['finetune_lr_tree'],
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=config['seed'], n_jobs=-1, verbose=-1
            )
            new_model.fit(X_h, y_h, init_model=base_model)
            new_models.append(new_model)

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        preds = np.column_stack([m.predict(X_test_scaled) for m in new_models])
        preds = target_scaler.inverse_transform(preds)
        return preds, target_scaler, new_models

    elif 'ann' in name_lower:
        model = nsw_model_dict['model']

        y_2d = np.full((len(y_target), horizon), np.nan)
        for i in range(horizon):
            if horizon - i > 0:
                y_2d[:-horizon+i, i] = y_target[horizon-i:, 0] if horizon-i > 0 else y_target[:, 0]
        valid = ~np.isnan(y_2d).any(axis=1)
        X_h = X_target_scaled[valid]
        # Transform each horizon column separately (scaler fitted on 1D)
        y_h = np.column_stack([
            target_scaler.transform(y_2d[valid, h].reshape(-1, 1)).flatten()
            for h in range(horizon)
        ])

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['finetune_lr'])
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_h).to(device)
        y_t = torch.FloatTensor(y_h).to(device)
        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=config['batch_size'], shuffle=True)

        model.train()
        for epoch in range(config['finetune_epochs']):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        X_test_t = torch.FloatTensor(X_test_scaled).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).cpu().numpy()
        pred = target_scaler.inverse_transform(pred)
        return pred, target_scaler, model

    else:
        # RNN full fine-tuning
        model = nsw_model_dict['model']

        y_target_flat = target_train_df['TOTALDEMAND'].values
        y_target_scaled = target_scaler.transform(y_target_flat.reshape(-1, 1)).flatten()
        X_seq, y_seq = create_sequences(X_target_scaled, y_target_scaled,
                                        config['lookback'], horizon)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['finetune_lr'])
        criterion = nn.MSELoss()

        X_t = torch.FloatTensor(X_seq).to(device)
        y_t = torch.FloatTensor(y_seq).to(device)
        if y_t.ndim == 1:
            y_t = y_t.reshape(-1, 1)

        loader = DataLoader(TensorDataset(X_t, y_t),
                           batch_size=config['batch_size'], shuffle=True)

        model.train()
        for epoch in range(config['finetune_epochs']):
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        X_test = target_test_df[feature_cols].fillna(0).values
        X_test_scaled = feature_scaler.transform(X_test)
        X_seq_test, _ = create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)),
                                         config['lookback'], horizon)
        X_test_t = torch.FloatTensor(X_seq_test).to(device)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test_t), config['batch_size']):
                batch = X_test_t[i:i+config['batch_size']]
                preds.append(model(batch).cpu().numpy())
        y_pred_scaled = np.vstack(preds)
        y_pred_scaled = np.clip(y_pred_scaled, -1.0, 1.0)
        orig_shape = y_pred_scaled.shape
        y_pred_flat = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_flat.reshape(orig_shape), target_scaler, model


# ============================================================
# Strategy 4: Few-shot Learning
# ============================================================

def strategy_fewshot(model_name, horizon, nsw_model_dict, target_train_df,
                     target_test_df, feature_cols, config, data_sizes_days,
                     target_region, save_models=True, transfer_model_dir=None):
    """
    Few-shot: run partial and full fine-tuning with different amounts of target data.
    """
    results = []
    for days in data_sizes_days:
        n_hours = days * 24
        if n_hours > len(target_train_df):
            continue
        subset = target_train_df.iloc[:n_hours].copy()

        # Partial
        try:
            pred_p, tsc_p, model_p = strategy_partial(
                model_name, horizon, nsw_model_dict,
                subset, target_test_df, feature_cols, config
            )
            y_true, pred_p, n_h = evaluate_predictions(
                model_name, horizon, pred_p, target_test_df, config)
            m_p = compute_metrics(y_true, pred_p, n_h, model_name, horizon,
                                  target_region, f'fewshot_partial_{days}d')
            results.append(m_p)

            if save_models and transfer_model_dir is not None:
                save_path = transfer_model_dir / (
                    f"{model_name.lower()}_{horizon}h_{target_region}_"
                    f"fewshot_partial_{days}d.pkl"
                )
                save_transfer_model(
                    model_p, nsw_model_dict['feature_scaler'], tsc_p,
                    {
                        'name': f'{nsw_model_dict["name"]}_fewshot_partial_{days}d',
                        'horizon': horizon,
                        'target_region': target_region,
                        'strategy': f'fewshot_partial_{days}d',
                        'model_type': model_name.lower(),
                    },
                    save_path
                )
        except Exception as e:
            print(f"    fewshot partial {days}d failed: {e}")

        # Full
        try:
            pred_f, tsc_f, model_f = strategy_full(
                model_name, horizon, nsw_model_dict,
                subset, target_test_df, feature_cols, config
            )
            y_true, pred_f, n_h = evaluate_predictions(
                model_name, horizon, pred_f, target_test_df, config)
            m_f = compute_metrics(y_true, pred_f, n_h, model_name, horizon,
                                  target_region, f'fewshot_full_{days}d')
            results.append(m_f)

            if save_models and transfer_model_dir is not None:
                save_path = transfer_model_dir / (
                    f"{model_name.lower()}_{horizon}h_{target_region}_"
                    f"fewshot_full_{days}d.pkl"
                )
                save_transfer_model(
                    model_f, nsw_model_dict['feature_scaler'], tsc_f,
                    {
                        'name': f'{nsw_model_dict["name"]}_fewshot_full_{days}d',
                        'horizon': horizon,
                        'target_region': target_region,
                        'strategy': f'fewshot_full_{days}d',
                        'model_type': model_name.lower(),
                    },
                    save_path
                )
        except Exception as e:
            print(f"    fewshot full {days}d failed: {e}")

    return results


# ============================================================
# Model Loading
# ============================================================

def load_nsw_model(model_path, device):
    """Load pickled model. Returns dict with model, scalers, name, horizon."""
    with open(model_path, 'rb') as f:
        save_dict = pickle.load(f)

    name = save_dict['name']
    model_type = name.lower()

    result = {
        'name': name,
        'horizon': save_dict['horizon'],
        'feature_scaler': save_dict['feature_scaler'],
        'target_scaler': save_dict['target_scaler'],
        'model': save_dict['model'],
        'device': device,
    }

    if 'ann' in model_type or 'lstm' in model_type or 'gru' in model_type:
        result['model'] = result['model'].to(device)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Transfer Learning for Cross-State STLF')
    parser.add_argument('--horizons', type=str, default='1,24,168')
    parser.add_argument('--models', type=str,
                        default='xgboost,lightgbm,ann,lstm,gru,bilstm,bigru')
    parser.add_argument('--target-regions', type=str, default='QLD1,SA1,TAS1,VIC1')
    parser.add_argument('--strategies', type=str, default='scaler,partial,full,fewshot',
                        help='scaler,partial,full,fewshot')
    parser.add_argument('--data-file', type=str,
                        default=f'aemo_features_{YEARS[0]}_{YEARS[-1]}.csv')
    parser.add_argument('--model-dir', type=str, default='./results/models')
    parser.add_argument('--output-dir', type=str, default='./results/transfer')
    parser.add_argument('--transfer-model-dir', type=str,
                        default='./results/transfer_models',
                        help='Directory to save fine-tuned transfer models')
    parser.add_argument('--save-models', action='store_true', default=True,
                        help='Save fine-tuned models (default: True)')
    parser.add_argument('--no-save-models', action='store_false', dest='save_models',
                        help='Do not save fine-tuned models')
    parser.add_argument('--test-year', type=int, default=YEARS[-1])
    parser.add_argument('--fewshot-days', type=str, default='30,90,180')
    parser.add_argument('--lookback', type=int, default=168)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                        help='Learning rate for PyTorch fine-tuning')
    parser.add_argument('--finetune-lr-tree', type=float, default=0.02,
                        help='Learning rate for tree fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'device': device,
        'lookback': args.lookback,
        'batch_size': args.batch_size,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'finetune_lr_tree': args.finetune_lr_tree,
        'seed': args.seed,
    }

    horizons = [int(h) for h in args.horizons.split(',')]
    models = args.models.split(',')
    target_regions = args.target_regions.split(',')
    strategies = args.strategies.split(',')
    fewshot_days = [int(d) for d in args.fewshot_days.split(',')]

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    transfer_model_dir = Path(args.transfer_model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        transfer_model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRANSFER LEARNING FOR CROSS-STATE STLF")
    print("=" * 80)
    print(f"Target regions: {target_regions}")
    print(f"Models: {models}")
    print(f"Horizons: {horizons}")
    print(f"Strategies: {strategies}")
    print(f"Save models: {args.save_models}")
    print(f"Transfer model dir: {transfer_model_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    all_results = []
    saved_count = 0

    for target_region in target_regions:
        print(f"\n{'='*80}")
        print(f"TARGET REGION: {target_region}")
        print(f"{'='*80}")

        loader = TargetDataLoader(args.data_file, target_region, args.test_year)
        target_train_df, target_test_df, feature_cols = loader.load()

        print(f"  [INFO] Target train: {len(target_train_df)} rows, "
              f"test: {len(target_test_df)} rows")
        print(f"  [INFO] Target demand range: "
              f"[{target_test_df['TOTALDEMAND'].min():.0f}, "
              f"{target_test_df['TOTALDEMAND'].max():.0f}] MW")

        for horizon in horizons:
            print(f"\n  Horizon: {horizon}h")

            for model_name in models:
                model_path = model_dir / f"{model_name.lower()}_{horizon}h.pkl"
                if not model_path.exists():
                    print(f"    [SKIP] {model_path} not found")
                    continue

                try:
                    nsw_dict = load_nsw_model(model_path, device)
                    print(f"    [MODEL] {model_name} loaded")
                except Exception as e:
                    print(f"    [ERROR] Failed to load {model_name}: {e}")
                    continue

                # ---- Strategy: scaler ----
                if 'scaler' in strategies:
                    try:
                        pred, tsc, model_obj = strategy_scaler(
                            model_name, horizon, nsw_dict,
                            target_train_df, target_test_df, feature_cols, config
                        )
                        y_true, pred, n_h = evaluate_predictions(
                            model_name, horizon, pred, target_test_df, config)
                        m = compute_metrics(y_true, pred, n_h, model_name, horizon,
                                            target_region, 'scaler')
                        all_results.append(m)
                        print(f"      scaler:  MAPE={m['MAPE']:.2f}%  R2={m['R2']:.2f}%")

                        if args.save_models:
                            save_path = transfer_model_dir / (
                                f"{model_name.lower()}_{horizon}h_{target_region}_scaler.pkl"
                            )
                            save_transfer_model(
                                model_obj, nsw_dict['feature_scaler'], tsc,
                                {
                                    'name': f'{nsw_dict["name"]}_scaler',
                                    'horizon': horizon,
                                    'target_region': target_region,
                                    'strategy': 'scaler',
                                    'model_type': model_name.lower(),
                                },
                                save_path
                            )
                            saved_count += 1
                    except Exception as e:
                        print(f"      scaler failed: {e}")

                # ---- Strategy: partial ----
                if 'partial' in strategies:
                    try:
                        pred, tsc, model_obj = strategy_partial(
                            model_name, horizon, nsw_dict,
                            target_train_df, target_test_df, feature_cols, config
                        )
                        y_true, pred, n_h = evaluate_predictions(
                            model_name, horizon, pred, target_test_df, config)
                        m = compute_metrics(y_true, pred, n_h, model_name, horizon,
                                            target_region, 'partial')
                        all_results.append(m)
                        print(f"      partial: MAPE={m['MAPE']:.2f}%  R2={m['R2']:.2f}%")

                        if args.save_models:
                            save_path = transfer_model_dir / (
                                f"{model_name.lower()}_{horizon}h_{target_region}_partial.pkl"
                            )
                            save_transfer_model(
                                model_obj, nsw_dict['feature_scaler'], tsc,
                                {
                                    'name': f'{nsw_dict["name"]}_partial',
                                    'horizon': horizon,
                                    'target_region': target_region,
                                    'strategy': 'partial',
                                    'model_type': model_name.lower(),
                                },
                                save_path
                            )
                            saved_count += 1
                    except Exception as e:
                        print(f"      partial failed: {e}")

                # ---- Strategy: full ----
                if 'full' in strategies:
                    try:
                        pred, tsc, model_obj = strategy_full(
                            model_name, horizon, nsw_dict,
                            target_train_df, target_test_df, feature_cols, config
                        )
                        y_true, pred, n_h = evaluate_predictions(
                            model_name, horizon, pred, target_test_df, config)
                        m = compute_metrics(y_true, pred, n_h, model_name, horizon,
                                            target_region, 'full')
                        all_results.append(m)
                        print(f"      full:    MAPE={m['MAPE']:.2f}%  R2={m['R2']:.2f}%")

                        if args.save_models:
                            save_path = transfer_model_dir / (
                                f"{model_name.lower()}_{horizon}h_{target_region}_full.pkl"
                            )
                            save_transfer_model(
                                model_obj, nsw_dict['feature_scaler'], tsc,
                                {
                                    'name': f'{nsw_dict["name"]}_full',
                                    'horizon': horizon,
                                    'target_region': target_region,
                                    'strategy': 'full',
                                    'model_type': model_name.lower(),
                                },
                                save_path
                            )
                            saved_count += 1
                    except Exception as e:
                        print(f"      full failed: {e}")

                # ---- Strategy: fewshot ----
                if 'fewshot' in strategies:
                    try:
                        few_results = strategy_fewshot(
                            model_name, horizon, nsw_dict,
                            target_train_df, target_test_df,
                            feature_cols, config, fewshot_days,
                            target_region,
                            save_models=args.save_models,
                            transfer_model_dir=transfer_model_dir if args.save_models else None
                        )
                        for m in few_results:
                            all_results.append(m)
                            print(f"      {m['Strategy']}: MAPE={m['MAPE']:.2f}%  R2={m['R2']:.2f}%")
                        if args.save_models:
                            saved_count += len(few_results)
                    except Exception as e:
                        print(f"      fewshot failed: {e}")

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'transfer_results.csv', index=False)

    if len(df) > 0:
        pivot = df.pivot_table(
            index=['Region', 'Model', 'Horizon'],
            columns='Strategy',
            values='MAPE'
        ).round(2)
        pivot.to_csv(output_dir / 'transfer_summary_mape.csv')

        print("\n" + "=" * 80)
        print("TRANSFER LEARNING SUMMARY (MAPE %)")
        print("=" * 80)
        print(pivot.to_string())
        print("=" * 80)

        plot_transfer_results(df, output_dir)

    print(f"\n[DONE] Results saved to {output_dir}")
    print(f"[DONE] Fine-tuned models saved to {transfer_model_dir} ({saved_count} files)")


def plot_transfer_results(df, output_dir):
    """Generate comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for region in df['Region'].unique():
        reg_df = df[df['Region'] == region]
        if len(reg_df) == 0:
            continue

        pivot = reg_df.pivot_table(
            index='Model', columns='Strategy', values='MAPE'
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
        ax.set_title(f'MAPE (%) - {region} (Lower is Better)')
        plt.tight_layout()
        plt.savefig(output_dir / f'transfer_heatmap_{region}.png', dpi=300)
        plt.close()

    fewshot_df = df[df['Strategy'].str.startswith('fewshot')]
    if len(fewshot_df) > 0:
        for region in fewshot_df['Region'].unique():
            reg_df = fewshot_df[fewshot_df['Region'] == region]
            fig, ax = plt.subplots(figsize=(10, 6))
            for model in reg_df['Model'].unique():
                for strategy_type in ['fewshot_partial', 'fewshot_full']:
                    sub = reg_df[(reg_df['Model'] == model) &
                                 (reg_df['Strategy'].str.contains(strategy_type))]
                    if len(sub) > 0:
                        days = sub['Strategy'].str.extract(r'(\d+)d').astype(int)[0]
                        ax.plot(days, sub['MAPE'].values, marker='o',
                                label=f'{model} ({strategy_type.replace("fewshot_", "")})')
            ax.set_xlabel('Target Data Size (days)')
            ax.set_ylabel('MAPE (%)')
            ax.set_title(f'Few-shot Learning Curve - {region}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'fewshot_curve_{region}.png', dpi=300)
            plt.close()

    print(f"  [SAVE] Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
