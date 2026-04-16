"""
Rule-based risk engine.

Provides a deterministic safety layer that can veto or override the LLM
decision, and computes position sizing / capital controls.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Rule engine (50% of ensemble weight)
# ---------------------------------------------------------------------------

def rule_engine_decision(
    market_feat: dict[str, Any],
    price_feat: dict[str, Any],
    sentiment_feat: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply hard-coded trading rules and return a rule-based decision.

    Parameters
    ----------
    market_feat : dict
        Output of ``compute_market_features``.
    price_feat : dict
        Output of ``compute_price_features``.
    sentiment_feat : dict
        Output of ``compute_sentiment``.

    Returns
    -------
    dict
        ``decision`` (str), ``confidence`` (float), ``risk_level`` (str).
    """
    regime = market_feat["market_regime"]
    in_buy_zone = price_feat["in_buy_zone"]
    sentiment = sentiment_feat["sentiment_label"]

    # Hard veto: never buy in RISK_OFF
    if regime == "RISK_OFF":
        return {
            "decision": "NO_TRADE",
            "confidence": 0.9,
            "reasoning": "Rule engine: RISK_OFF regime — no buy.",
            "risk_level": "HIGH",
        }

    if in_buy_zone and sentiment == "POSITIVE":
        return {
            "decision": "BUY",
            "confidence": 0.7,
            "reasoning": "Rule engine: price in value zone with positive sentiment.",
            "risk_level": "LOW",
        }

    if in_buy_zone:
        return {
            "decision": "WATCH",
            "confidence": 0.5,
            "reasoning": "Rule engine: price in value zone but sentiment not positive.",
            "risk_level": "MEDIUM",
        }

    return {
        "decision": "NO_TRADE",
        "confidence": 0.6,
        "reasoning": "Rule engine: price outside value zone.",
        "risk_level": "MEDIUM",
    }


# ---------------------------------------------------------------------------
# Ensemble combiner
# ---------------------------------------------------------------------------

def ensemble_decision(
    rule_result: dict[str, Any],
    llm_result: dict[str, Any],
    rule_weight: float = 0.5,
    llm_weight: float = 0.5,
) -> dict[str, Any]:
    """
    Combine rule-engine and LLM decisions using weighted voting.

    Each decision maps to a numeric score:
      BUY=1, WATCH=0, NO_TRADE=-1.

    The weighted sum is thresholded to produce a final decision.

    Parameters
    ----------
    rule_result : dict
        Decision dict from ``rule_engine_decision``.
    llm_result : dict
        Decision dict from ``parse_llm_json``.
    rule_weight : float
        Weight of the rule engine (default 0.5).
    llm_weight : float
        Weight of the LLM output (default 0.5).

    Returns
    -------
    dict
        Final blended decision with keys: ``decision``, ``confidence``,
        ``reasoning``, ``risk_level``.
    """
    score_map = {"BUY": 1, "WATCH": 0, "NO_TRADE": -1}

    rule_score = score_map.get(rule_result["decision"], 0)
    llm_score = score_map.get(llm_result["decision"], 0)

    combined_score = rule_weight * rule_score + llm_weight * llm_score
    combined_confidence = (
        rule_weight * rule_result["confidence"]
        + llm_weight * llm_result["confidence"]
    )

    if combined_score > 0.3:
        final_decision = "BUY"
    elif combined_score > -0.3:
        final_decision = "WATCH"
    else:
        final_decision = "NO_TRADE"

    # Use the higher risk level of the two
    risk_priority = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    risk_levels = [rule_result.get("risk_level", "MEDIUM"), llm_result.get("risk_level", "MEDIUM")]
    final_risk = max(risk_levels, key=lambda r: risk_priority.get(r, 1))

    return {
        "decision": final_decision,
        "confidence": round(combined_confidence, 4),
        "reasoning": (
            f"Ensemble (rule={rule_result['decision']}, "
            f"llm={llm_result['decision']}, score={combined_score:.2f}). "
            f"LLM: {llm_result.get('reasoning', '')}"
        ),
        "risk_level": final_risk,
    }


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    decision: str,
    confidence: float,
    risk_level: str,
    portfolio_value: float = 100_000.0,
    max_risk_pct: float = 0.02,
) -> float:
    """
    Compute the dollar amount to deploy based on the final decision.

    Parameters
    ----------
    decision : str
        ``"BUY"``, ``"WATCH"``, or ``"NO_TRADE"``.
    confidence : float
        Confidence score in [0, 1].
    risk_level : str
        ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
    portfolio_value : float
        Total portfolio value in dollars (default 100 000).
    max_risk_pct : float
        Maximum fraction of portfolio to risk per trade (default 2 %).

    Returns
    -------
    float
        Dollar position size (0 for WATCH / NO_TRADE).
    """
    if decision != "BUY":
        return 0.0

    risk_multipliers = {"LOW": 1.0, "MEDIUM": 0.6, "HIGH": 0.2}
    risk_mult = risk_multipliers.get(risk_level, 0.5)

    position = portfolio_value * max_risk_pct * confidence * risk_mult
    return round(position, 2)
