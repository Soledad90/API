"""
Main LLM-driven trading agent.

Architecture (Cognitive Trading):

    Market Data → Feature Engineering → LLM + Rule Engine → Ensemble
                                      → Risk Engine → Position Size
                                      → Memory / Feedback Loop

Multi-agent sub-system
----------------------
- MacroAgent    : evaluates market regime
- TechnicalAgent: evaluates price structure
- SentimentAgent: evaluates news / sentiment
- DecisionAgent : synthesises all signals into a final decision
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .data import fetch_news, get_market_data
from .features import compute_market_features, compute_price_features, compute_sentiment
from .llm import call_llm, parse_llm_json
from .memory import (
    compute_performance_stats,
    feedback_confidence_adjustment,
    load_memory,
    save_memory,
)
from .prompt import build_prompt
from .risk import compute_position_size, ensemble_decision, rule_engine_decision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual specialist agents
# ---------------------------------------------------------------------------

class MacroAgent:
    """Evaluates market regime from breadth data."""

    def analyse(self, market_data: dict[str, Any]) -> dict[str, Any]:
        features = compute_market_features(market_data)
        logger.debug("MacroAgent features: %s", features)
        return features


class TechnicalAgent:
    """Evaluates price structure relative to a value zone."""

    def analyse(self, price: float) -> dict[str, Any]:
        features = compute_price_features(price)
        logger.debug("TechnicalAgent features: %s", features)
        return features


class SentimentAgent:
    """Evaluates news sentiment."""

    def analyse(self, headlines: list[str]) -> dict[str, Any]:
        features = compute_sentiment(headlines)
        logger.debug("SentimentAgent features: %s", features)
        return features


class DecisionAgent:
    """
    Synthesises macro, technical, and sentiment signals.

    Uses an LLM as the primary reasoning core (50 % weight) combined with
    a deterministic rule engine (50 % weight) to produce a blended decision.
    """

    def decide(
        self,
        market_feat: dict[str, Any],
        price_feat: dict[str, Any],
        sentiment_feat: dict[str, Any],
        price: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Return (llm_result, rule_result).

        Parameters
        ----------
        market_feat, price_feat, sentiment_feat : dict
            Feature dicts from the specialist agents.
        price : float
            Current asset price.

        Returns
        -------
        tuple[dict, dict]
            (llm_result, rule_result) — both in the standard decision format.
        """
        # Rule engine (deterministic)
        rule_result = rule_engine_decision(market_feat, price_feat, sentiment_feat)

        # LLM reasoning core
        prompt = build_prompt(market_feat, price_feat, sentiment_feat, price)
        raw_llm = call_llm(prompt)
        llm_result = parse_llm_json(raw_llm)

        return llm_result, rule_result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_agent_llm(
    portfolio_value: float = 100_000.0,
    persist_memory: bool = True,
) -> dict[str, Any]:
    """
    Run the full LLM-driven trading agent pipeline.

    Steps
    -----
    1. Fetch market data and news.
    2. Run specialist agents (Macro, Technical, Sentiment).
    3. Run the Decision agent (LLM + Rule engine).
    4. Blend results via ensemble combiner.
    5. Apply feedback-loop confidence adjustment.
    6. Compute position size via the risk engine.
    7. Persist the trade record to memory.

    Parameters
    ----------
    portfolio_value : float
        Total portfolio value used for position sizing.
    persist_memory : bool
        Whether to write the trade record to the memory file.

    Returns
    -------
    dict
        Final blended decision, position size, and performance stats.
    """
    # ---- 1. Data ----
    market_data = get_market_data()
    headlines = fetch_news()

    # ---- 2. Specialist agents ----
    macro_agent = MacroAgent()
    technical_agent = TechnicalAgent()
    sentiment_agent = SentimentAgent()

    market_feat = macro_agent.analyse(market_data)
    price_feat = technical_agent.analyse(market_data["price"])
    sentiment_feat = sentiment_agent.analyse(headlines)

    # ---- 3 & 4. Decision + ensemble ----
    decision_agent = DecisionAgent()
    llm_result, rule_result = decision_agent.decide(
        market_feat, price_feat, sentiment_feat, market_data["price"]
    )

    final = ensemble_decision(rule_result, llm_result)

    # ---- 5. Feedback loop ----
    history = load_memory()
    final["confidence"] = feedback_confidence_adjustment(
        final["confidence"], history
    )

    # ---- 6. Position sizing ----
    position_size = compute_position_size(
        final["decision"],
        final["confidence"],
        final["risk_level"],
        portfolio_value=portfolio_value,
    )
    final["position_size_usd"] = position_size

    # ---- 7. Memory ----
    trade_record = {
        "price": market_data["price"],
        "market_feat": market_feat,
        "price_feat": price_feat,
        "sentiment_feat": sentiment_feat,
        "rule_result": rule_result,
        "llm_result": llm_result,
        "final_decision": final,
        # outcome is filled in later when the trade is closed
        "outcome": None,
        "confidence": final["confidence"],
    }
    if persist_memory:
        save_memory(trade_record)

    # ---- Output ----
    stats = compute_performance_stats(history)

    output = {
        "decision": final,
        "performance": stats,
    }

    print("\n🧠 AI DECISION:")
    print(json.dumps(final, indent=2))
    print("\n📊 PERFORMANCE STATS:")
    print(json.dumps(stats, indent=2))

    return output
