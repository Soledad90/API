"""
Tests for the memory / feedback-loop module.
"""

import json
import os
import tempfile
import pytest

from src.memory import (
    load_memory,
    save_memory,
    compute_performance_stats,
    feedback_confidence_adjustment,
)


@pytest.fixture
def tmp_memory_file(tmp_path):
    return str(tmp_path / "trade_history.json")


class TestLoadSaveMemory:
    def test_load_returns_empty_list_when_file_missing(self, tmp_memory_file):
        assert load_memory(tmp_memory_file) == []

    def test_save_and_load_round_trip(self, tmp_memory_file):
        record = {"decision": "BUY", "confidence": 0.8, "outcome": "WIN"}
        save_memory(record, tmp_memory_file)
        history = load_memory(tmp_memory_file)
        assert len(history) == 1
        assert history[0]["decision"] == "BUY"

    def test_multiple_saves_append(self, tmp_memory_file):
        save_memory({"outcome": "WIN"}, tmp_memory_file)
        save_memory({"outcome": "LOSS"}, tmp_memory_file)
        assert len(load_memory(tmp_memory_file)) == 2

    def test_timestamp_added_automatically(self, tmp_memory_file):
        save_memory({"decision": "WATCH"}, tmp_memory_file)
        history = load_memory(tmp_memory_file)
        assert "timestamp" in history[0]


class TestComputePerformanceStats:
    def test_empty_history(self):
        stats = compute_performance_stats([])
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

    def test_all_wins(self):
        history = [{"outcome": "WIN", "confidence": 0.8}] * 5
        stats = compute_performance_stats(history)
        assert stats["wins"] == 5
        assert stats["win_rate"] == pytest.approx(1.0)

    def test_mixed_outcomes(self):
        history = [
            {"outcome": "WIN", "confidence": 0.9},
            {"outcome": "LOSS", "confidence": 0.6},
            {"outcome": "WIN", "confidence": 0.7},
        ]
        stats = compute_performance_stats(history)
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=1e-3)


class TestFeedbackConfidenceAdjustment:
    def test_no_history_returns_base(self):
        assert feedback_confidence_adjustment(0.8, []) == pytest.approx(0.8)

    def test_all_losses_reduces_confidence(self):
        history = [{"outcome": "LOSS"}] * 10
        adjusted = feedback_confidence_adjustment(0.8, history)
        assert adjusted < 0.8

    def test_all_wins_unchanged_confidence(self):
        history = [{"outcome": "WIN"}] * 10
        adjusted = feedback_confidence_adjustment(0.8, history)
        assert adjusted == pytest.approx(0.8)

    def test_result_clamped_to_zero_on_extreme_losses(self):
        history = [{"outcome": "LOSS"}] * 100
        adjusted = feedback_confidence_adjustment(0.3, history)
        assert adjusted >= 0.0
