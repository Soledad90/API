"""
Tests for the LLM module (parse_llm_json).

call_llm() itself is not tested here because it requires a live API key.
"""

import json
import pytest
from src.llm import parse_llm_json


class TestParseLlmJson:
    def _make_raw(self, data: dict) -> str:
        return json.dumps(data)

    def test_valid_buy_decision(self):
        raw = self._make_raw({
            "decision": "BUY",
            "confidence": 0.85,
            "reasoning": "Good conditions.",
            "risk_level": "LOW",
        })
        result = parse_llm_json(raw)
        assert result["decision"] == "BUY"
        assert result["confidence"] == pytest.approx(0.85)
        assert result["risk_level"] == "LOW"

    def test_valid_no_trade_decision(self):
        raw = self._make_raw({
            "decision": "NO_TRADE",
            "confidence": 0.9,
            "reasoning": "RISK_OFF regime.",
            "risk_level": "HIGH",
        })
        result = parse_llm_json(raw)
        assert result["decision"] == "NO_TRADE"

    def test_valid_watch_decision(self):
        raw = self._make_raw({
            "decision": "WATCH",
            "confidence": 0.5,
            "reasoning": "Mixed signals.",
            "risk_level": "MEDIUM",
        })
        result = parse_llm_json(raw)
        assert result["decision"] == "WATCH"

    def test_strips_markdown_fences(self):
        inner = json.dumps({
            "decision": "BUY",
            "confidence": 0.7,
            "reasoning": "OK.",
            "risk_level": "LOW",
        })
        raw = f"```json\n{inner}\n```"
        result = parse_llm_json(raw)
        assert result["decision"] == "BUY"

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError, match="valid JSON"):
            parse_llm_json("not json at all")

    def test_missing_key_raises_value_error(self):
        raw = self._make_raw({
            "decision": "BUY",
            "confidence": 0.7,
            # missing reasoning and risk_level
        })
        with pytest.raises(ValueError, match="missing required keys"):
            parse_llm_json(raw)

    def test_invalid_decision_raises_value_error(self):
        raw = self._make_raw({
            "decision": "SELL",
            "confidence": 0.7,
            "reasoning": "Test.",
            "risk_level": "LOW",
        })
        with pytest.raises(ValueError, match="Invalid decision value"):
            parse_llm_json(raw)

    def test_invalid_risk_level_raises_value_error(self):
        raw = self._make_raw({
            "decision": "BUY",
            "confidence": 0.7,
            "reasoning": "Test.",
            "risk_level": "CRITICAL",
        })
        with pytest.raises(ValueError, match="Invalid risk_level"):
            parse_llm_json(raw)

    def test_confidence_out_of_range_raises_value_error(self):
        raw = self._make_raw({
            "decision": "BUY",
            "confidence": 1.5,
            "reasoning": "Test.",
            "risk_level": "LOW",
        })
        with pytest.raises(ValueError, match="confidence must be between"):
            parse_llm_json(raw)
