"""
Tests for the risk engine module.
"""

import pytest
from src.risk import rule_engine_decision, ensemble_decision, compute_position_size


class TestRuleEngineDecision:
    def _market(self, regime="NEUTRAL"):
        return {"market_regime": regime, "breadth_ratio": 1.0}

    def _price(self, in_zone=True, discount=0.02):
        return {"in_buy_zone": in_zone, "discount_to_value": discount}

    def _sentiment(self, label="POSITIVE"):
        return {"sentiment_label": label, "sentiment_score": 0.5}

    def test_risk_off_always_no_trade(self):
        result = rule_engine_decision(
            self._market("RISK_OFF"), self._price(True), self._sentiment("POSITIVE")
        )
        assert result["decision"] == "NO_TRADE"
        assert result["risk_level"] == "HIGH"

    def test_neutral_in_zone_positive_sentiment_is_buy(self):
        result = rule_engine_decision(
            self._market("NEUTRAL"), self._price(True), self._sentiment("POSITIVE")
        )
        assert result["decision"] == "BUY"
        assert result["risk_level"] == "LOW"

    def test_neutral_in_zone_negative_sentiment_is_watch(self):
        result = rule_engine_decision(
            self._market("NEUTRAL"), self._price(True), self._sentiment("NEGATIVE")
        )
        assert result["decision"] == "WATCH"

    def test_neutral_outside_zone_is_no_trade(self):
        result = rule_engine_decision(
            self._market("NEUTRAL"), self._price(False), self._sentiment("POSITIVE")
        )
        assert result["decision"] == "NO_TRADE"


class TestEnsembleDecision:
    def _dec(self, decision, confidence, risk_level="MEDIUM"):
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": "test",
            "risk_level": risk_level,
        }

    def test_both_buy_gives_buy(self):
        result = ensemble_decision(self._dec("BUY", 0.8), self._dec("BUY", 0.7))
        assert result["decision"] == "BUY"

    def test_both_no_trade_gives_no_trade(self):
        result = ensemble_decision(
            self._dec("NO_TRADE", 0.9), self._dec("NO_TRADE", 0.8)
        )
        assert result["decision"] == "NO_TRADE"

    def test_conflicting_signals_gives_watch(self):
        result = ensemble_decision(
            self._dec("BUY", 0.7), self._dec("NO_TRADE", 0.7)
        )
        assert result["decision"] == "WATCH"

    def test_higher_risk_propagates(self):
        result = ensemble_decision(
            self._dec("BUY", 0.8, "LOW"), self._dec("BUY", 0.8, "HIGH")
        )
        assert result["risk_level"] == "HIGH"

    def test_confidence_is_blended(self):
        result = ensemble_decision(
            self._dec("BUY", 0.8), self._dec("BUY", 0.6)
        )
        assert result["confidence"] == pytest.approx(0.7)


class TestComputePositionSize:
    def test_no_trade_gives_zero(self):
        assert compute_position_size("NO_TRADE", 0.9, "LOW") == 0.0

    def test_watch_gives_zero(self):
        assert compute_position_size("WATCH", 0.9, "LOW") == 0.0

    def test_buy_low_risk_gives_nonzero(self):
        size = compute_position_size("BUY", 1.0, "LOW", portfolio_value=100_000)
        assert size > 0

    def test_buy_high_risk_smaller_than_low_risk(self):
        low = compute_position_size("BUY", 1.0, "LOW", portfolio_value=100_000)
        high = compute_position_size("BUY", 1.0, "HIGH", portfolio_value=100_000)
        assert high < low

    def test_lower_confidence_gives_smaller_size(self):
        big = compute_position_size("BUY", 1.0, "LOW", portfolio_value=100_000)
        small = compute_position_size("BUY", 0.5, "LOW", portfolio_value=100_000)
        assert small < big
