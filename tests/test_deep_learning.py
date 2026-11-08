"""Focused tests for Phase 5 deterministic deep-learning building blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))

from deep_learning import (  # noqa: E402
    DeepLearningConfig,
    FeatureNormalizer,
    LSTMSequenceClassifier,
    MLPClassifier,
    TCNSequenceClassifier,
    build_sequence_dataset,
    classifier_scores,
    train_classifier,
)


def test_normalizer_uses_train_statistics_and_supports_sequence_broadcasting():
    normalizer = FeatureNormalizer().fit(np.array([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32))
    transformed = normalizer.transform(np.array([[[1.0, 10.0], [3.0, 14.0]]], dtype=np.float32))
    assert transformed.shape == (1, 2, 2)
    assert np.allclose(transformed[0, 0], [-1.0, -1.0])
    assert np.allclose(transformed[0, 1], [1.0, 1.0])


def test_sequence_windows_end_on_target_day_without_future_rows():
    frame = pd.DataFrame({
        "merchant_id": ["A", "A", "A", "B", "B", "B"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"] * 2,
        "has_sufficient_history": [True] * 6,
        "label_fraud_spike": [0, 1, 0, 1, 0, 1],
        "feature_a": [1, 2, 3, 10, 11, 12],
    })
    sequences = build_sequence_dataset(frame, ["feature_a"], sequence_length=2)
    assert sequences.values.shape == (4, 2, 1)
    assert sequences.values[0, :, 0].tolist() == [1, 2]
    assert sequences.labels.tolist() == [1, 0, 0, 1]
    assert pd.Timestamp(sequences.target_dates[0]).date().isoformat() == "2026-01-02"


def test_neural_classifiers_return_one_logit_per_row_or_sequence():
    tabular = MLPClassifier(3, 12, 0.0)
    lstm = LSTMSequenceClassifier(3, 12, 0.0)
    tcn = TCNSequenceClassifier(3, 12, 0.0)
    assert tabular(torch.ones(2, 3)).shape == (2,)
    assert lstm(torch.ones(2, 4, 3)).shape == (2,)
    assert tcn(torch.ones(2, 4, 3)).shape == (2,)


def test_classifier_training_uses_weighted_loss_and_writes_restorable_checkpoint(tmp_path):
    config = DeepLearningConfig(max_epochs=3, patience=2, batch_size=2, hidden_dim=8, dropout=0.0)
    model = MLPClassifier(2, config.hidden_dim, config.dropout)
    train_values = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], dtype=np.float32)
    train_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    validation_values = np.array([[0, 0], [2, 2], [1, 1], [2, 1]], dtype=np.float32)
    validation_labels = np.array([0, 1, 1, 1], dtype=np.int64)
    checkpoint = tmp_path / "mlp.pt"
    report = train_classifier(
        model, train_values, train_labels, validation_values, validation_labels, config, str(checkpoint)
    )
    assert checkpoint.is_file()
    assert report["positive_class_weight"] == 1.0
    assert 1 <= report["best_epoch"] <= config.max_epochs
    scores = classifier_scores(model, validation_values)
    assert np.all((scores >= 0) & (scores <= 1))
