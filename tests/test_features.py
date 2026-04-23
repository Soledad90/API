"""
Tests for feature engineering module.
"""

import pytest
from src.features import (
    compute_market_features,
    compute_price_features,
    compute_sentiment,
    sentiment_score,
)


class TestComputeMarketFeatures:
    def test_risk_off_when_breadth_ratio_above_3(self):
        data = {"advance": 100, "decline": 400}
        result = compute_market_features(data)
        assert result["market_regime"] == "RISK_OFF"
        assert result["breadth_ratio"] == pytest.approx(4.0)

    def test_neutral_when_breadth_ratio_below_3(self):
        data = {"advance": 200, "decline": 300}
        result = compute_market_features(data)
        assert result["market_regime"] == "NEUTRAL"
        assert result["breadth_ratio"] == pytest.approx(1.5)

    def test_zero_advance_uses_max_of_1(self):
        data = {"advance": 0, "decline": 500}
        result = compute_market_features(data)
        assert result["breadth_ratio"] == pytest.approx(500.0)
        assert result["market_regime"] == "RISK_OFF"

    def test_equal_advance_decline(self):
        data = {"advance": 300, "decline": 300}
        result = compute_market_features(data)
        assert result["breadth_ratio"] == pytest.approx(1.0)
        assert result["market_regime"] == "NEUTRAL"


class TestComputePriceFeatures:
    def test_price_inside_buy_zone(self):
        result = compute_price_features(17_500.0)
        assert result["in_buy_zone"] is True

    def test_price_below_buy_zone(self):
        result = compute_price_features(16_000.0)
        assert result["in_buy_zone"] is False

    def test_price_above_buy_zone(self):
        result = compute_price_features(19_000.0)
        assert result["in_buy_zone"] is False

    def test_price_at_lower_bound_is_in_zone(self):
        result = compute_price_features(17_000.0)
        assert result["in_buy_zone"] is True

    def test_price_at_upper_bound_is_in_zone(self):
        result = compute_price_features(18_000.0)
        assert result["in_buy_zone"] is True

    def test_discount_to_value_calculation(self):
        result = compute_price_features(17_100.0)
        expected = (18_000 - 17_100) / 18_000
        assert result["discount_to_value"] == pytest.approx(expected, rel=1e-4)

    def test_negative_discount_when_above_value_high(self):
        result = compute_price_features(19_000.0)
        assert result["discount_to_value"] < 0


class TestSentimentScore:
    def test_empty_headlines_returns_zero(self):
        assert sentiment_score([]) == 0.0

    def test_positive_headline(self):
        score = sentiment_score(["Markets rally on strong earnings."])
        assert score > 0

    def test_negative_headline(self):
        score = sentiment_score(["Markets fall amid uncertainty."])
        assert score < 0

    def test_neutral_headline(self):
        score = sentiment_score(["The market opened today."])
        assert score == 0.0


class TestComputeSentiment:
    def test_positive_label_for_positive_score(self):
        result = compute_sentiment(["Stocks surge on bullish data."])
        assert result["sentiment_label"] == "POSITIVE"
        assert result["sentiment_score"] > 0

    def test_negative_label_for_negative_score(self):
        result = compute_sentiment(["Markets decline amid tensions."])
        assert result["sentiment_label"] == "NEGATIVE"
        assert result["sentiment_score"] < 0

    def test_zero_score_is_negative_label(self):
        result = compute_sentiment(["The market opened today."])
        assert result["sentiment_label"] == "NEGATIVE"
        assert result["sentiment_score"] == 0.0
