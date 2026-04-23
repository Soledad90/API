"""
Rule-based risk engine — VN Stock Market.

Provides a deterministic safety layer that can veto or override the LLM
decision, and computes position sizing / capital controls.

VN-specific rules:
  - Biên độ giao động: ±7% (HOSE)
  - Không mua khi RSI > 75 (overbought nặng)
  - Không mua khi volume thấp bất thường
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
    tech_feat: dict[str, Any] | None = None,
    smc_feat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply hard-coded trading rules including SMC/ICT institutional logic.
    """
    regime = market_feat.get("market_regime", "NEUTRAL")
    in_buy_zone = price_feat.get("in_buy_zone", False)
    sentiment = sentiment_feat.get("sentiment_label", "NEUTRAL")

    # Extract tech/smc signals
    tech = tech_feat or {}
    smc = smc_feat or {}
    
    rsi = tech.get("rsi", 50.0) or 50.0
    ma_signal = tech.get("ma_signal", "UNKNOWN")
    vol_signal = tech.get("vol_signal", "NORMAL")
    
    struct = smc.get("structure", "NEUTRAL")
    sweep = smc.get("liquidity_sweep")
    trading_mode = smc.get("trading_mode", "SWING")

    # ---- Rule 1: Institutional VETO — No Trend/No Sweep ----
    # Bluechip strategy: No BUY if HTF is bearish and no liquidity sweep
    if regime == "RISK_OFF":
        return {
            "decision": "NO_TRADE",
            "confidence": 0.90,
            "reasoning": "Quy tắc: Market RISK_OFF — Vĩ mô không thuận lợi.",
            "risk_level": "HIGH",
        }

    # ---- Rule 2: Institutional BUY — The 'Spring' (Sweep + CHoCH) ----
    if sweep == "BULLISH_SWEEP" and struct in ("BULLISH", "BULLISH_REVERSAL"):
        return {
            "decision": "BUY",
            "confidence": 0.85,
            "reasoning": f"Quy tắc [{trading_mode}]: Đã có quét thanh khoản Bullish Sweep + Xác nhận đổi chiều cấu trúc.",
            "risk_level": "LOW",
        }

    # ---- Rule 3: SELL signal — Sweep + Structure Break ----
    if sweep == "BEARISH_SWEEP" or (rsi > 75 and struct == "BEARISH"):
        return {
            "decision": "SELL",
            "confidence": 0.80,
            "reasoning": "Quy tắc: Bearish Sweep hoặc RSI quá mua kèm cấu trúc giảm.",
            "risk_level": "HIGH",
        }

    # ---- Rule 4: Trend Following BULLISH (BOS) ----
    if smc.get("bos") == "BULLISH_BOS" and not smc.get("premium_zone"):
        return {
            "decision": "BUY",
            "confidence": 0.75,
            "reasoning": "Quy tắc: Tiếp diễn cấu trúc tăng (BOS) và đang ở vùng giá DISCOUNT.",
            "risk_level": "LOW",
        }

    # ---- Rule 5: Basic TA Fallback ----
    if in_buy_zone and sentiment == "POSITIVE" and ma_signal in ("BULLISH", "GOLDEN_CROSS"):
        return {
            "decision": "BUY",
            "confidence": 0.65,
            "reasoning": "Quy tắc: Thuận xu hướng kỹ thuật cơ bản trong vùng giá trị.",
            "risk_level": "MEDIUM",
        }

    # ---- Default: Wait for Signal ----
    if sweep is None and struct == "NEUTRAL":
        return {
            "decision": "WATCH",
            "confidence": 0.50,
            "reasoning": "Quy tắc: Chưa có tín hiệu quét thanh khoản hoặc phá vỡ cấu trúc — Chờ đợi.",
            "risk_level": "MEDIUM",
        }

    return {
        "decision": "NO_TRADE",
        "confidence": 0.60,
        "reasoning": "Quy tắc: Không tìm thấy sự hợp lưu (Confluence) giữa SMC và TA.",
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

    Score map: BUY=1, WATCH=0, NO_TRADE=-1, SELL=-2.

    Parameters
    ----------
    rule_result : dict
        Decision dict from ``rule_engine_decision``.
    llm_result : dict
        Decision dict from ``parse_llm_json``.
    rule_weight, llm_weight : float
        Weights (default 50/50).

    Returns
    -------
    dict
        Final blended decision.
    """
    score_map = {"BUY": 1, "WATCH": 0, "NO_TRADE": -1, "SELL": -2}

    rule_score = score_map.get(rule_result.get("decision", "NO_TRADE"), -1)
    llm_score = score_map.get(llm_result.get("decision", "NO_TRADE"), -1)

    combined_score = rule_weight * rule_score + llm_weight * llm_score
    combined_confidence = (
        rule_weight * rule_result.get("confidence", 0.5)
        + llm_weight * llm_result.get("confidence", 0.5)
    )

    if combined_score >= 0.8:
        final_decision = "BUY"
    elif combined_score >= 0.1:
        final_decision = "WATCH"
    elif combined_score >= -0.8:
        final_decision = "NO_TRADE"
    else:
        final_decision = "SELL"

    risk_priority = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
    risk_levels = [
        rule_result.get("risk_level", "MEDIUM"),
        llm_result.get("risk_level", "MEDIUM"),
    ]
    final_risk = max(risk_levels, key=lambda r: risk_priority.get(r, 1))

    return {
        "decision": final_decision,
        "confidence": round(combined_confidence, 4),
        "reasoning": (
            f"Ensemble (rule={rule_result.get('decision')}, "
            f"llm={llm_result.get('decision')}, score={combined_score:.2f}). "
            f"LLM: {llm_result.get('reasoning', '')}"
        ),
        "risk_level": final_risk,
        "key_signals": llm_result.get("key_signals", []),
    }


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_position_size(
    decision: str,
    confidence: float,
    risk_level: str,
    portfolio_value: float = 100_000_000.0,   # 100 triệu VND mặc định
    max_risk_pct: float = 0.02,               # 2% rủi ro tối đa / lệnh
) -> float:
    """
    Compute the VND amount to deploy based on the final decision.

    Parameters
    ----------
    decision : str
        ``"BUY"``, ``"SELL"``, ``"WATCH"``, or ``"NO_TRADE"``.
    confidence : float
        Confidence score in [0, 1].
    risk_level : str
        ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``.
    portfolio_value : float
        Total portfolio value in VND (default 100,000,000 VND).
    max_risk_pct : float
        Maximum fraction of portfolio to risk per trade (default 2%).

    Returns
    -------
    float
        VND position size (0 for WATCH / NO_TRADE;
        negative for SELL to indicate reduce/exit).
    """
    if decision in ("WATCH", "NO_TRADE"):
        return 0.0

    risk_multipliers = {"LOW": 1.0, "MEDIUM": 0.6, "HIGH": 0.25}
    risk_mult = risk_multipliers.get(risk_level, 0.5)

    position = portfolio_value * max_risk_pct * confidence * risk_mult

    if decision == "SELL":
        return -round(position, 0)   # âm = giảm vị thế
    return round(position, 0)
