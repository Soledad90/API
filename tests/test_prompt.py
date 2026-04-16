"""
Tests for the prompt builder module.
"""

from src.prompt import build_prompt


class TestBuildPrompt:
    def _market_feat(self):
        return {"breadth_ratio": 1.5, "market_regime": "NEUTRAL"}

    def _price_feat(self):
        return {"in_buy_zone": True, "discount_to_value": 0.0278}

    def _sentiment_feat(self):
        return {"sentiment_score": 0.5, "sentiment_label": "POSITIVE"}

    def test_returns_string(self):
        prompt = build_prompt(
            self._market_feat(), self._price_feat(), self._sentiment_feat(), 17_500.0
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_regime(self):
        prompt = build_prompt(
            self._market_feat(), self._price_feat(), self._sentiment_feat(), 17_500.0
        )
        assert "NEUTRAL" in prompt

    def test_contains_price(self):
        prompt = build_prompt(
            self._market_feat(), self._price_feat(), self._sentiment_feat(), 17_500.0
        )
        assert "17500.0" in prompt or "17500" in prompt

    def test_contains_output_format(self):
        prompt = build_prompt(
            self._market_feat(), self._price_feat(), self._sentiment_feat(), 17_500.0
        )
        assert "decision" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "risk_level" in prompt.lower()

    def test_risk_off_regime_in_prompt(self):
        market_feat = {"breadth_ratio": 4.0, "market_regime": "RISK_OFF"}
        prompt = build_prompt(
            market_feat, self._price_feat(), self._sentiment_feat(), 17_500.0
        )
        assert "RISK_OFF" in prompt
