"""
Feature engineering layer.

Converts raw market data and news into structured signals that are fed
to the prompt builder and the rule-based engine.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def compute_market_features(data: dict[str, Any]) -> dict[str, Any]:
    """
    Derive market-breadth features from advance/decline data.

    Parameters
    ----------
    data : dict
        Must contain ``advance`` (int) and ``decline`` (int) keys.

    Returns
    -------
    dict
        - ``breadth_ratio`` (float): decline / advance ratio.
        - ``market_regime`` (str): ``"RISK_OFF"`` when breadth_ratio > 3,
          else ``"NEUTRAL"``.
    """
    advance = data["advance"]
    decline = data["decline"]

    breadth_ratio = decline / max(advance, 1)

    return {
        "breadth_ratio": round(breadth_ratio, 4),
        "market_regime": "RISK_OFF" if breadth_ratio > 3 else "NEUTRAL",
    }


# ---------------------------------------------------------------------------
# Price structure
# ---------------------------------------------------------------------------

def compute_price_features(
    price: float,
    value_low: float = 17_000.0,
    value_high: float = 18_000.0,
) -> dict[str, Any]:
    """
    Derive price-structure features relative to a value zone.

    Parameters
    ----------
    price : float
        Current asset price.
    value_low : float
        Lower bound of the fair-value zone (default 17 000).
    value_high : float
        Upper bound of the fair-value zone (default 18 000).

    Returns
    -------
    dict
        - ``in_buy_zone`` (bool): True when price is within the value zone.
        - ``discount_to_value`` (float): (value_high - price) / value_high,
          positive means price is below the zone ceiling.
    """
    in_buy_zone = value_low <= price <= value_high
    discount_to_value = (value_high - price) / value_high

    return {
        "in_buy_zone": in_buy_zone,
        "discount_to_value": round(discount_to_value, 6),
    }


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

def sentiment_score(headlines: list[str]) -> float:
    """
    Compute a simple sentiment score from a list of news headlines.

    Returns a value in [-1, 1].  Positive words contribute +1/n,
    negative words contribute -1/n (where n = number of headlines).

    Parameters
    ----------
    headlines : list[str]
        News headline strings.

    Returns
    -------
    float
        Aggregate normalised sentiment score.

    Notes
    -----
    Replace with a proper NLP model (e.g. FinBERT) in production.
    """
    positive_words = {
        "rally", "surge", "gain", "strong", "bullish", "positive",
        "growth", "beat", "up", "rise", "rises",
    }
    negative_words = {
        "fall", "drop", "weak", "bearish", "negative", "loss", "miss",
        "down", "decline", "tensions", "weigh", "uncertainty",
    }

    if not headlines:
        return 0.0

    score = 0.0
    for headline in headlines:
        words = set(headline.lower().split())
        score += len(words & positive_words)
        score -= len(words & negative_words)

    return round(score / len(headlines), 4)


def compute_sentiment(headlines: list[str]) -> dict[str, Any]:
    """
    Build a sentiment feature dict from raw news headlines.

    Parameters
    ----------
    headlines : list[str]
        News headline strings.

    Returns
    -------
    dict
        - ``sentiment_score`` (float): aggregate normalised score (positive/negative word counts per headline).
        - ``sentiment_label`` (str): ``"POSITIVE"`` or ``"NEGATIVE"``.
    """
    score = sentiment_score(headlines)

    return {
        "sentiment_score": score,
        "sentiment_label": "POSITIVE" if score > 0 else "NEGATIVE",
    }
