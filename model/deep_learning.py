"""Deterministic PyTorch building blocks for offline Phase 5 experiments only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DeepLearningConfig:
    seed: int = 42
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 48
    dropout: float = 0.10
    sequence_length: int = 14

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceDataset:
    values: np.ndarray
    labels: np.ndarray
    target_dates: np.ndarray
    merchant_ids: np.ndarray


def set_deterministic(seed: int) -> None:
    """Set all supported CPU random sources; experiments intentionally use CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


class FeatureNormalizer:
    """Train-only standardization usable for tabular and sequence tensors."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "FeatureNormalizer":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not len(values):
            raise ValueError("Normalizer fit requires non-empty two-dimensional values.")
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale = np.where(self.scale < 1e-6, 1.0, self.scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit the feature normalizer before transforming values.")
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean) / self.scale).astype(np.float32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class NormalBehaviorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        latent = max(4, hidden_dim // 4)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(values)
        return self.head(hidden[-1]).squeeze(-1)


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.convolution = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=self.padding)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.convolution(values)
        output = output[:, :, :-self.padding]
        return values + self.dropout(self.activation(output))


class TCNSequenceClassifier(nn.Module):
    """Small causal temporal convolutional network using the same sequence inputs as LSTM."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(_CausalBlock(hidden_dim, 1, dropout), _CausalBlock(hidden_dim, 2, dropout))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        channels_first = values.transpose(1, 2)
        encoded = self.blocks(self.input_projection(channels_first))
        return self.head(encoded[:, :, -1]).squeeze(-1)


def build_sequence_dataset(
    featured: pd.DataFrame,
    feature_columns: list[str],
    *,
    sequence_length: int,
) -> SequenceDataset:
    """Build prior-and-current feature windows ending on each predicted day.

    Features themselves are end-of-day, leakage-safe values. A sequence ending
    at day T contains only the merchant's observed rows through T and never
    labels or feature rows from a later date.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two.")
    required = {"merchant_id", "date", "label_fraud_spike", "has_sufficient_history", *feature_columns}
    missing = sorted(required - set(featured.columns))
    if missing:
        raise ValueError(f"Sequence construction lacks fields: {missing}")
    usable = featured[featured["has_sufficient_history"]].copy()
    usable["date"] = pd.to_datetime(usable["date"], errors="raise")
    windows: list[np.ndarray] = []
    labels: list[int] = []
    dates: list[np.datetime64] = []
    merchants: list[str] = []
    for merchant_id, group in usable.sort_values(["merchant_id", "date"]).groupby("merchant_id", sort=False):
        group = group.sort_values("date")
        values = group[feature_columns].to_numpy(dtype=np.float32)
        for end in range(sequence_length - 1, len(group)):
            windows.append(values[end - sequence_length + 1:end + 1])
            labels.append(int(group.iloc[end]["label_fraud_spike"]))
            dates.append(group.iloc[end]["date"].to_datetime64())
            merchants.append(str(merchant_id))
    if not windows:
        raise ValueError("No merchant has enough observed history for the requested sequence length.")
    return SequenceDataset(
        values=np.stack(windows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        target_dates=np.asarray(dates),
        merchant_ids=np.asarray(merchants),
    )


def _loader(values: np.ndarray, labels: np.ndarray | None, config: DeepLearningConfig, *, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(values, dtype=np.float32))]
    if labels is not None:
        tensors.append(torch.from_numpy(np.asarray(labels, dtype=np.float32)))
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(TensorDataset(*tensors), batch_size=config.batch_size, shuffle=shuffle, generator=generator)


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_classifier(
    model: nn.Module,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    config: DeepLearningConfig,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Train a binary classifier with weighted loss, early stopping, and checkpointing."""
    set_deterministic(config.seed)
    positives = int(np.asarray(train_labels).sum())
    negatives = len(train_labels) - positives
    if not positives or not negatives:
        raise ValueError("Classifier training requires both classes.")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_epoch, remaining = float("inf"), 0, config.patience
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_rows = 0
        for values, labels in _loader(train_values, train_labels, config, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(values)
            train_rows += len(values)
        model.eval()
        validation_loss = 0.0
        validation_rows = 0
        with torch.no_grad():
            for values, labels in _loader(validation_values, validation_labels, config, shuffle=False):
                loss = criterion(model(values), labels)
                validation_loss += float(loss.item()) * len(values)
                validation_rows += len(values)
        validation_loss /= validation_rows
        history.append({"epoch": epoch, "train_loss": round(train_loss / train_rows, 8), "validation_loss": round(validation_loss, 8)})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, remaining = validation_loss, epoch, config.patience
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss}, checkpoint_path)
        else:
            remaining -= 1
            if remaining == 0:
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": round(float(best_loss), 8),
        "epochs_ran": len(history),
        "positive_class_weight": round(float(negatives / positives), 6),
        "history": history,
        "checkpoint_path": checkpoint_path,
        "checkpoint_state_sha256": _state_hash(model),
    }


def train_autoencoder(
    model: nn.Module,
    normal_train_values: np.ndarray,
    normal_validation_values: np.ndarray,
    config: DeepLearningConfig,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Train a normal-only reconstruction model with early stopping/checkpointing."""
    set_deterministic(config.seed)
    if not len(normal_train_values) or not len(normal_validation_values):
        raise ValueError("Autoencoder requires normal train and validation observations.")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_loss, best_epoch, remaining = float("inf"), 0, config.patience
    history: list[dict[str, float]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        rows = 0
        for (values,) in _loader(normal_train_values, None, config, shuffle=True):
            optimizer.zero_grad()
            loss = criterion(model(values), values)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(values)
            rows += len(values)
        model.eval()
        with torch.no_grad():
            validation_values = torch.from_numpy(np.asarray(normal_validation_values, dtype=np.float32))
            validation_loss = float(criterion(model(validation_values), validation_values).item())
        history.append({"epoch": epoch, "train_loss": round(train_loss / rows, 8), "validation_loss": round(validation_loss, 8)})
        if validation_loss < best_loss - 1e-8:
            best_loss, best_epoch, remaining = validation_loss, epoch, config.patience
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": validation_loss}, checkpoint_path)
        else:
            remaining -= 1
            if remaining == 0:
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_validation_loss": round(float(best_loss), 8),
        "epochs_ran": len(history),
        "normal_training_rows": int(len(normal_train_values)),
        "checkpoint_path": checkpoint_path,
        "checkpoint_state_sha256": _state_hash(model),
    }


def classifier_scores(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(values, dtype=np.float32)))
    return torch.sigmoid(logits).cpu().numpy().astype(float)


def reconstruction_errors(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    with torch.no_grad():
        reconstructed = model(tensor)
    return ((reconstructed - tensor).pow(2).mean(dim=1)).cpu().numpy().astype(float)


def measure_latency_ms(model: nn.Module, values: np.ndarray, *, repeats: int = 15) -> dict[str, float]:
    """CPU inference latency with warm-up, reported per input row/sequence."""
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
    with torch.no_grad():
        for _ in range(3):
            model(tensor)
        started = time.perf_counter()
        for _ in range(repeats):
            model(tensor)
        elapsed = time.perf_counter() - started
    return {
        "batch_rows": int(len(tensor)),
        "repeats": repeats,
        "mean_batch_ms": round(elapsed * 1000 / repeats, 6),
        "mean_per_row_ms": round(elapsed * 1000 / (repeats * len(tensor)), 6),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
