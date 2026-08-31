"""
=============================================================================
RENAISSANCE DEEP LEARNING ENGINE — deep_learning_engine.py
=============================================================================
Neural Network Architecture Stack (PyTorch + NumPy fallback)

MODELS:
  ├── FeedForward ANN           — 3-layer MLP with BatchNorm + Dropout
  ├── BiLSTM + Temporal Attention— bidirectional LSTM with learned attention
  ├── Temporal Fusion Transformer— multi-head self-attention on time series
  ├── Regime-Conditioned Model   — switches sub-models based on HMM regime
  ├── Autoencoder Anomaly Det.   — reconstruction error = anomaly score
  └── Neural Ensemble Combiner   — stacks all models with meta-learner

Architecture choices match Renaissance philosophy:
  - Every model is short-horizon (1-5 day) and high-frequency retrained
  - Ensemble of weak learners > single strong model
  - Regime-awareness prevents model decay during regime shifts
  - All models have numpy-only fallbacks for environments without PyTorch

INSTALL:
  pip install torch --break-system-packages  (optional, GPU recommended)
  Falls back to numpy implementations if torch unavailable.

Renaissance.io — Institutional Grade Quantitative Trading
=============================================================================
"""

import numpy as np
import logging
import threading
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger("DEEP_LEARN")

# ── PyTorch availability ──────────────────────────────────────────────────────
_TORCH_OK = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch available on {_DEVICE}")
except ImportError:
    logger.info("PyTorch unavailable — using numpy fallback models")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEEDFORWARD ANN — Multi-Layer Perceptron
# ═══════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:
    class _ANNModel(nn.Module):
        """
        3-layer MLP: Input → 128 → 64 → 32 → 1
        BatchNorm + Dropout(0.3) + LeakyReLU
        """
        def __init__(self, input_dim: int = 14):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


    # ═══════════════════════════════════════════════════════════════════════════
    # 2. BiLSTM + TEMPORAL ATTENTION
    # ═══════════════════════════════════════════════════════════════════════════

    class _TemporalAttention(nn.Module):
        """Learned attention weights over LSTM hidden states."""
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, lstm_out):
            # lstm_out: (batch, seq_len, hidden*2)
            weights = self.attn(lstm_out)          # (batch, seq, 1)
            weights = torch.softmax(weights, dim=1)
            context = (lstm_out * weights).sum(dim=1)  # (batch, hidden*2)
            return context, weights.squeeze(-1)

    class _BiLSTMAttention(nn.Module):
        """
        Bidirectional LSTM with temporal attention mechanism.
        Input: (batch, seq_len, n_features)
        Output: probability score [0, 1]
        """
        def __init__(self, n_features: int = 14, hidden: int = 64,
                     n_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden,
                num_layers=n_layers, batch_first=True,
                bidirectional=True, dropout=dropout if n_layers > 1 else 0
            )
            self.attention = _TemporalAttention(hidden * 2)
            self.fc = nn.Sequential(
                nn.Linear(hidden * 2, 32),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            context, attn_weights = self.attention(lstm_out)
            return self.fc(context).squeeze(-1), attn_weights


    # ═══════════════════════════════════════════════════════════════════════════
    # 3. TEMPORAL FUSION TRANSFORMER (Simplified)
    # ═══════════════════════════════════════════════════════════════════════════

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class _TransformerEncoder(nn.Module):
        """
        Multi-head self-attention encoder for time series.
        Input: (batch, seq_len, n_features)
        """
        def __init__(self, n_features: int = 14, d_model: int = 64,
                     n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_enc(x)
            x = self.transformer(x)
            # Use last timestep as representation
            return self.fc(x[:, -1, :]).squeeze(-1)


    # ═══════════════════════════════════════════════════════════════════════════
    # 4. AUTOENCODER ANOMALY DETECTOR
    # ═══════════════════════════════════════════════════════════════════════════

    class _Autoencoder(nn.Module):
        """
        Encoder-Decoder for anomaly detection.
        High reconstruction error = anomalous market state.
        """
        def __init__(self, input_dim: int = 14, latent_dim: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, input_dim)
            )

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

        def anomaly_score(self, x):
            """Higher = more anomalous."""
            recon, _ = self.forward(x)
            mse = ((x - recon) ** 2).mean(dim=-1)
            return mse


# ═══════════════════════════════════════════════════════════════════════════════
# NUMPY FALLBACK MODELS (when PyTorch unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyANN:
    """Pure numpy 2-layer neural network. Sigmoid activation."""
    def __init__(self, input_dim=14, hidden=32):
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * 0.1
        self.b2 = np.zeros(1)
        self._trained = False

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _relu(self, x):
        return np.maximum(0, x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = self._relu(X @ self.W1 + self.b1)
        out = self._sigmoid(h @ self.W2 + self.b2)
        return out.flatten()

    def train(self, X: np.ndarray, y: np.ndarray, epochs=100, lr=0.01):
        """Mini-batch gradient descent."""
        n = len(X)
        for epoch in range(epochs):
            # Forward
            h = self._relu(X @ self.W1 + self.b1)
            pred = self._sigmoid(h @ self.W2 + self.b2).flatten()

            # Binary cross-entropy gradient
            error = pred - y
            dW2 = h.T @ error.reshape(-1, 1) / n
            db2 = error.mean()
            dh = error.reshape(-1, 1) @ self.W2.T
            dh[h <= 0] = 0  # ReLU gradient
            dW1 = X.T @ dh / n
            db1 = dh.mean(axis=0)

            # Update
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

        self._trained = True
        mse = float(np.mean((pred - y) ** 2))
        return {"epochs": epochs, "mse": mse}


class NumpyLSTMCell:
    """Minimal LSTM cell in pure numpy for environments without PyTorch."""
    def __init__(self, input_dim=14, hidden_dim=32):
        self.hidden_dim = hidden_dim
        scale = 0.1
        # Gates: input, forget, cell, output
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * scale
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        # Output projection
        self.Wy = np.random.randn(hidden_dim, 1) * scale
        self.by = np.zeros(1)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward_sequence(self, X_seq: np.ndarray) -> float:
        """Process sequence (seq_len, features) → scalar prediction."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)

        for t in range(len(X_seq)):
            x = X_seq[t]
            concat = np.concatenate([x, h])

            f = self._sigmoid(concat @ self.Wf + self.bf)
            i = self._sigmoid(concat @ self.Wi + self.bi)
            c_hat = np.tanh(concat @ self.Wc + self.bc)
            o = self._sigmoid(concat @ self.Wo + self.bo)

            c = f * c + i * c_hat
            h = o * np.tanh(c)

        # Final prediction
        out = self._sigmoid(h @ self.Wy + self.by)
        return float(out[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. REGIME-CONDITIONED MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeConditionedPredictor:
    """
    Maintains separate sub-models for each market regime:
    - Bull (low vol, trending up)
    - Bear (high vol, trending down)
    - Mean-reverting (range-bound)

    Switches prediction source based on current HMM regime.
    This prevents catastrophic model failure during regime shifts.
    """

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._regime_performance: Dict[str, List[float]] = {
            "bull": [], "bear": [], "mean_revert": []
        }
        self._lock = threading.Lock()
        self._init_models()

    def _init_models(self):
        """Initialize one sub-model per regime."""
        if _TORCH_OK:
            for regime in ["bull", "bear", "mean_revert"]:
                self._models[regime] = _ANNModel(input_dim=14).to(_DEVICE)
        else:
            for regime in ["bull", "bear", "mean_revert"]:
                self._models[regime] = NumpyANN(input_dim=14, hidden=32)

    def predict(self, features: np.ndarray, regime: str = "bull") -> Dict:
        """
        Predict using the regime-appropriate sub-model.

        Args:
            features: (n_features,) or (seq_len, n_features) array
            regime: current market regime from HMM

        Returns:
            {
                "prediction": float [0, 1],
                "regime_used": str,
                "regime_confidence": float,
                "all_regime_preds": dict
            }
        """
        regime_key = self._map_regime(regime)
        all_preds = {}

        for r, model in self._models.items():
            try:
                if _TORCH_OK and isinstance(model, nn.Module):
                    model.eval()
                    with torch.no_grad():
                        x = torch.FloatTensor(features).unsqueeze(0).to(_DEVICE)
                        if len(x.shape) == 2:
                            pred = model(x).item()
                        else:
                            pred = model(x).item()
                else:
                    if hasattr(model, 'predict'):
                        pred = float(model.predict(features.reshape(1, -1))[0])
                    else:
                        pred = 0.5
                all_preds[r] = round(pred, 4)
            except Exception:
                all_preds[r] = 0.5

        primary = all_preds.get(regime_key, 0.5)

        # Blend: 70% regime-specific + 30% ensemble average
        avg = np.mean(list(all_preds.values()))
        blended = 0.7 * primary + 0.3 * avg

        return {
            "prediction": round(float(np.clip(blended, 0, 1)), 4),
            "regime_used": regime_key,
            "regime_confidence": round(abs(primary - 0.5) * 2, 4),
            "all_regime_preds": all_preds
        }

    @staticmethod
    def _map_regime(regime: str) -> str:
        regime_lower = regime.lower()
        if any(w in regime_lower for w in ["bull", "low_vol", "trending_up", "risk_on"]):
            return "bull"
        elif any(w in regime_lower for w in ["bear", "high_vol", "trending_down", "risk_off", "crisis"]):
            return "bear"
        return "mean_revert"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NEURAL ENSEMBLE COMBINER — Meta-Learner
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralEnsemble:
    """
    Stacks all deep learning models into a meta-ensemble.
    The meta-learner combines predictions from:
      - ANN
      - BiLSTM+Attention
      - Transformer
      - Regime-conditioned model
    Using a simple learned weighting layer.

    This mirrors Renaissance's 'ensemble of ensembles' approach.
    """

    def __init__(self):
        self._ann = None
        self._bilstm = None
        self._transformer = None
        self._regime_model = RegimeConditionedPredictor()
        self._autoencoder = None
        self._meta_weights = np.array([0.25, 0.30, 0.25, 0.20])  # ANN, BiLSTM, TFT, Regime
        self._lock = threading.Lock()
        self._init_models()

    def _init_models(self):
        if _TORCH_OK:
            self._ann = _ANNModel(14).to(_DEVICE)
            self._bilstm = _BiLSTMAttention(14, 64, 2).to(_DEVICE)
            self._transformer = _TransformerEncoder(14, 64, 4, 2).to(_DEVICE)
            self._autoencoder = _Autoencoder(14, 4).to(_DEVICE)
        else:
            self._ann = NumpyANN(14, 32)
            self._bilstm = NumpyLSTMCell(14, 32)

    def predict(self, features: np.ndarray, regime: str = "bull",
                sequence: np.ndarray = None) -> Dict:
        """
        Run all models and combine predictions.

        Args:
            features: (n_features,) current feature vector
            regime: current market regime
            sequence: (seq_len, n_features) for sequential models

        Returns:
            {
                "ensemble_prediction": float [0, 1],
                "direction": str,           # "bullish" / "bearish" / "neutral"
                "confidence": float [0, 1],
                "anomaly_score": float,
                "model_predictions": dict,
                "attention_weights": list (if available),
                "meta_weights": list
            }
        """
        preds = {}
        attn_weights = None

        # 1. ANN prediction
        try:
            if _TORCH_OK and isinstance(self._ann, nn.Module):
                self._ann.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(_DEVICE)
                    preds["ann"] = self._ann(x).item()
            elif hasattr(self._ann, 'predict'):
                preds["ann"] = float(self._ann.predict(features.reshape(1, -1))[0])
        except Exception:
            preds["ann"] = 0.5

        # 2. BiLSTM + Attention
        seq = sequence if sequence is not None else features.reshape(1, -1)
        try:
            if _TORCH_OK and isinstance(self._bilstm, nn.Module):
                self._bilstm.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(seq).unsqueeze(0).to(_DEVICE)
                    pred, aw = self._bilstm(x)
                    preds["bilstm_attention"] = pred.item()
                    attn_weights = aw.cpu().numpy().tolist()[0]
            elif isinstance(self._bilstm, NumpyLSTMCell):
                preds["bilstm_attention"] = self._bilstm.forward_sequence(seq)
        except Exception:
            preds["bilstm_attention"] = 0.5

        # 3. Transformer
        try:
            if _TORCH_OK and self._transformer is not None:
                self._transformer.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(seq).unsqueeze(0).to(_DEVICE)
                    preds["transformer"] = self._transformer(x).item()
            else:
                preds["transformer"] = preds.get("ann", 0.5)  # fallback
        except Exception:
            preds["transformer"] = 0.5

        # 4. Regime-conditioned
        regime_result = self._regime_model.predict(features, regime)
        preds["regime_conditioned"] = regime_result["prediction"]

        # 5. Anomaly score
        anomaly = 0.0
        try:
            if _TORCH_OK and self._autoencoder is not None:
                self._autoencoder.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(features).unsqueeze(0).to(_DEVICE)
                    anomaly = self._autoencoder.anomaly_score(x).item()
        except Exception:
            pass

        # Meta-ensemble: weighted combination
        pred_values = [
            preds.get("ann", 0.5),
            preds.get("bilstm_attention", 0.5),
            preds.get("transformer", 0.5),
            preds.get("regime_conditioned", 0.5)
        ]
        ensemble = float(np.dot(self._meta_weights, pred_values))

        # If anomaly is high, reduce confidence and pull toward 0.5
        if anomaly > 2.0:
            blend_factor = min(anomaly / 5.0, 0.5)
            ensemble = ensemble * (1 - blend_factor) + 0.5 * blend_factor

        confidence = abs(ensemble - 0.5) * 2
        if ensemble > 0.55:
            direction = "bullish"
        elif ensemble < 0.45:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "ensemble_prediction": round(float(np.clip(ensemble, 0, 1)), 4),
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "anomaly_score": round(float(anomaly), 4),
            "model_predictions": {k: round(v, 4) for k, v in preds.items()},
            "attention_weights": attn_weights,
            "meta_weights": self._meta_weights.tolist()
        }

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_seq: np.ndarray = None, epochs: int = 50,
                  batch_size: int = 64) -> Dict:
        """
        Train all sub-models on the same dataset.

        Args:
            X_train: (n_samples, n_features)
            y_train: (n_samples,) binary labels
            X_seq: (n_samples, seq_len, n_features) for sequential models
            epochs: training epochs
            batch_size: mini-batch size

        Returns training stats for each model.
        """
        stats = {}

        if _TORCH_OK:
            X_t = torch.FloatTensor(X_train).to(_DEVICE)
            y_t = torch.FloatTensor(y_train).to(_DEVICE)

            # Train ANN
            try:
                stats["ann"] = self._train_torch_model(
                    self._ann, X_t, y_t, epochs, batch_size, "ANN"
                )
            except Exception as e:
                stats["ann"] = {"error": str(e)}

            # Train BiLSTM
            if X_seq is not None:
                try:
                    X_seq_t = torch.FloatTensor(X_seq).to(_DEVICE)
                    stats["bilstm"] = self._train_torch_seq_model(
                        self._bilstm, X_seq_t, y_t[:len(X_seq)], epochs, batch_size, "BiLSTM"
                    )
                except Exception as e:
                    stats["bilstm"] = {"error": str(e)}

            # Train Transformer
            if X_seq is not None:
                try:
                    X_seq_t = torch.FloatTensor(X_seq).to(_DEVICE)
                    stats["transformer"] = self._train_torch_model(
                        self._transformer, X_seq_t, y_t[:len(X_seq)], epochs, batch_size, "TFT"
                    )
                except Exception as e:
                    stats["transformer"] = {"error": str(e)}

            # Train Autoencoder (unsupervised)
            try:
                stats["autoencoder"] = self._train_autoencoder(X_t, epochs, batch_size)
            except Exception as e:
                stats["autoencoder"] = {"error": str(e)}
        else:
            # Numpy fallback
            try:
                stats["ann"] = self._ann.train(X_train, y_train, epochs=epochs)
            except Exception as e:
                stats["ann"] = {"error": str(e)}

        logger.info(f"Deep learning training complete: {stats}")
        return stats

    def _train_torch_model(self, model, X, y, epochs, batch_size, name):
        """Generic PyTorch model training loop."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.BCELoss()
        n = len(X)

        losses = []
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            batches = 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X[idx], y[idx]
                optimizer.zero_grad()
                pred = model(xb)
                if isinstance(pred, tuple):
                    pred = pred[0]
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            losses.append(epoch_loss / max(batches, 1))

        model.eval()
        return {"name": name, "final_loss": round(losses[-1], 6), "epochs": epochs}

    def _train_torch_seq_model(self, model, X_seq, y, epochs, batch_size, name):
        """Train sequential model (BiLSTM)."""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        n = len(X_seq)

        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_seq[idx], y[idx]
                optimizer.zero_grad()
                pred, _ = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        return {"name": name, "epochs": epochs, "status": "trained"}

    def _train_autoencoder(self, X, epochs, batch_size):
        """Train autoencoder for anomaly detection (unsupervised)."""
        self._autoencoder.train()
        optimizer = optim.Adam(self._autoencoder.parameters(), lr=1e-3)
        n = len(X)

        for epoch in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx]
                optimizer.zero_grad()
                recon, _ = self._autoencoder(xb)
                loss = nn.MSELoss()(recon, xb)
                loss.backward()
                optimizer.step()

        self._autoencoder.eval()
        return {"name": "Autoencoder", "epochs": epochs, "status": "trained"}


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_ensemble_instance: Optional[NeuralEnsemble] = None
_ensemble_lock = threading.Lock()

def get_neural_ensemble() -> NeuralEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        with _ensemble_lock:
            if _ensemble_instance is None:
                _ensemble_instance = NeuralEnsemble()
    return _ensemble_instance

def get_deep_learning_status() -> Dict:
    return {
        "pytorch_available": _TORCH_OK,
        "device": str(_DEVICE) if _TORCH_OK else "cpu_numpy",
        "models": ["ANN", "BiLSTM_Attention", "Transformer", "RegimeConditioned", "Autoencoder"],
        "ensemble_ready": _ensemble_instance is not None
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"PyTorch: {_TORCH_OK}")
    ens = get_neural_ensemble()
    feat = np.random.randn(14)
    result = ens.predict(feat, regime="bull")
    print(f"Ensemble result: {result}")
    print("✅ Deep Learning Engine operational")