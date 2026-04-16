"""
Prompt engineering for the LLM trading agent.

The prompt is the most critical component of the system – it encodes the
domain knowledge and trading rules that guide the LLM's reasoning.
"""

from __future__ import annotations

from typing import Any


def build_prompt(
    market_feat: dict[str, Any],
    price_feat: dict[str, Any],
    sentiment_feat: dict[str, Any],
    price: float,
) -> str:
    """
    Build a hedge-fund-grade prompt for the LLM decision engine.

    Parameters
    ----------
    market_feat : dict
        Output of ``compute_market_features``.
    price_feat : dict
        Output of ``compute_price_features``.
    sentiment_feat : dict
        Output of ``compute_sentiment``.
    price : float
        Current asset price (raw value, for readability in the prompt).

    Returns
    -------
    str
        Fully formatted prompt string ready to be sent to the LLM.
    """
    prompt = f"""You are a professional hedge fund trader.

MARKET CONDITIONS:
- Breadth Ratio: {market_feat['breadth_ratio']}
- Regime: {market_feat['market_regime']}

PRICE:
- Current Price: {price}
- In Buy Zone: {price_feat['in_buy_zone']}
- Discount to Value: {price_feat['discount_to_value']}

SENTIMENT:
- Score: {sentiment_feat['sentiment_score']}
- Label: {sentiment_feat['sentiment_label']}

RULES:
1. DO NOT BUY in RISK_OFF regime.
2. Only BUY if price is in the value zone.
3. Consider sentiment but do not overweight it.
4. WATCH means monitor but do not act.
5. NO_TRADE means conditions are unfavourable.

OUTPUT FORMAT (STRICT JSON — no markdown, no extra text):
{{
    "decision": "BUY | NO_TRADE | WATCH",
    "confidence": <float 0-1>,
    "reasoning": "<short explanation>",
    "risk_level": "LOW | MEDIUM | HIGH"
}}"""

    return prompt
