"""
Feature engineering layer — VN Stock Market.

Converts raw OHLCV + breadth data into structured technical signals:
  - Market regime    : breadth, trend strength
  - Technical        : MA cross, RSI, MACD, Bollinger Bands, Volume
  - Sentiment        : keyword-based scoring trên tiếng Việt & tiếng Anh
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market regime (breadth)
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
        - ``breadth_ratio`` (float)
        - ``market_regime`` (str): ``"RISK_OFF"`` | ``"NEUTRAL"`` | ``"RISK_ON"``
        - ``breadth_pct`` (float): advance/(advance+decline)
    """
    advance = max(int(data.get("advance", 1)), 1)
    decline = max(int(data.get("decline", 1)), 1)
    total = advance + decline

    breadth_ratio = decline / advance
    breadth_pct = advance / total

    if breadth_ratio > 2.5:
        regime = "RISK_OFF"
    elif breadth_pct > 0.55:
        regime = "RISK_ON"
    else:
        regime = "NEUTRAL"

    return {
        "breadth_ratio": round(breadth_ratio, 4),
        "breadth_pct": round(breadth_pct, 4),
        "market_regime": regime,
    }


# ---------------------------------------------------------------------------
# Technical indicators (from OHLCV DataFrame)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _macd(close: pd.Series) -> tuple[float, float, float]:
    """Return (macd_line, signal_line, histogram)."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return (
        round(float(macd_line.iloc[-1]), 4),
        round(float(signal.iloc[-1]), 4),
        round(float(hist.iloc[-1]), 4),
    )


def _bollinger(close: pd.Series, period: int = 20) -> dict[str, float]:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    price = close.iloc[-1]
    bw = (upper.iloc[-1] - lower.iloc[-1]) / ma.iloc[-1]   # bandwidth
    pct_b = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9)
    return {
        "bb_upper": round(float(upper.iloc[-1]), 2),
        "bb_mid": round(float(ma.iloc[-1]), 2),
        "bb_lower": round(float(lower.iloc[-1]), 2),
        "bb_bandwidth": round(float(bw), 4),
        "bb_pct_b": round(float(pct_b), 4),   # 0=lower band, 1=upper band
    }


def compute_technical_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute full suite of technical indicators from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
        Rows sorted oldest → newest.

    Returns
    -------
    dict with keys:
        ma5, ma20, ma50, ma_signal,
        rsi, rsi_zone,
        macd, macd_signal, macd_hist, macd_cross,
        bb_upper, bb_mid, bb_lower, bb_bandwidth, bb_pct_b,
        vol_ratio, vol_signal,
        price_change_pct, trend
    """
    if df is None or len(df) < 26:
        logger.warning("Không đủ dữ liệu cho indicators (< 26 bars)")
        return _empty_indicators()

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # ---- Moving Averages ----
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50_series = close.rolling(50).mean()
    ma50 = ma50_series.iloc[-1] if len(close) >= 50 else float("nan")
    price = close.iloc[-1]

    # Golden/Death cross detection (MA5 vs MA20)
    prev_ma5 = close.rolling(5).mean().iloc[-2]
    prev_ma20 = close.rolling(20).mean().iloc[-2]
    if prev_ma5 <= prev_ma20 and ma5 > ma20:
        ma_signal = "GOLDEN_CROSS"
    elif prev_ma5 >= prev_ma20 and ma5 < ma20:
        ma_signal = "DEATH_CROSS"
    elif ma5 > ma20:
        ma_signal = "BULLISH"
    else:
        ma_signal = "BEARISH"

    # ---- RSI ----
    rsi = _rsi(close)
    if rsi >= 70:
        rsi_zone = "OVERBOUGHT"
    elif rsi <= 30:
        rsi_zone = "OVERSOLD"
    else:
        rsi_zone = "NEUTRAL"

    # ---- MACD ----
    macd_val, macd_sig, macd_hist = _macd(close)
    prev_macd = float((_ema(close, 12) - _ema(close, 26)).iloc[-2])
    prev_sig = float(_ema(_ema(close, 12) - _ema(close, 26), 9).iloc[-2])
    if prev_macd <= prev_sig and macd_val > macd_sig:
        macd_cross = "BULLISH_CROSS"
    elif prev_macd >= prev_sig and macd_val < macd_sig:
        macd_cross = "BEARISH_CROSS"
    elif macd_val > macd_sig:
        macd_cross = "ABOVE_SIGNAL"
    else:
        macd_cross = "BELOW_SIGNAL"

    # ---- Bollinger Bands ----
    bb = _bollinger(close)

    # ---- Volume ----
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    vol_ratio = float(volume.iloc[-1]) / (vol_ma20 + 1e-9)
    if vol_ratio > 2.0:
        vol_signal = "HIGH_VOLUME"
    elif vol_ratio > 1.3:
        vol_signal = "ABOVE_AVERAGE"
    elif vol_ratio < 0.5:
        vol_signal = "LOW_VOLUME"
    else:
        vol_signal = "NORMAL"

    # ---- Price change ----
    price_change_pct = (close.iloc[-1] - close.iloc[-2]) / (close.iloc[-2] + 1e-9) * 100

    # ---- Overall trend ----
    bullish_signals = sum([
        ma_signal in ("GOLDEN_CROSS", "BULLISH"),
        rsi_zone == "OVERSOLD",
        macd_cross in ("BULLISH_CROSS", "ABOVE_SIGNAL"),
        bb["bb_pct_b"] < 0.2,    # gần lower band — potential buy
        vol_ratio > 1.3 and price_change_pct > 0,
    ])
    if bullish_signals >= 4:
        trend = "STRONG_BUY"
    elif bullish_signals >= 3:
        trend = "BUY"
    elif bullish_signals <= 1:
        trend = "SELL"
    else:
        trend = "NEUTRAL"

    return {
        # MAs
        "ma5": round(float(ma5), 2),
        "ma20": round(float(ma20), 2),
        "ma50": round(float(ma50), 2) if not np.isnan(ma50) else None,
        "ma_signal": ma_signal,
        # RSI
        "rsi": rsi,
        "rsi_zone": rsi_zone,
        # MACD
        "macd": macd_val,
        "macd_signal_line": macd_sig,
        "macd_hist": macd_hist,
        "macd_cross": macd_cross,
        # Bollinger
        **bb,
        # Volume
        "vol_ratio": round(vol_ratio, 2),
        "vol_signal": vol_signal,
        # Summary
        "price_change_pct": round(float(price_change_pct), 2),
        "trend": trend,
        "bullish_count": bullish_signals,
    }


def _empty_indicators() -> dict[str, Any]:
    return {
        "ma5": None, "ma20": None, "ma50": None, "ma_signal": "UNKNOWN",
        "rsi": 50.0, "rsi_zone": "NEUTRAL",
        "macd": 0.0, "macd_signal_line": 0.0, "macd_hist": 0.0, "macd_cross": "UNKNOWN",
        "bb_upper": None, "bb_mid": None, "bb_lower": None,
        "bb_bandwidth": None, "bb_pct_b": 0.5,
        "vol_ratio": 1.0, "vol_signal": "NORMAL",
        "price_change_pct": 0.0, "trend": "NEUTRAL", "bullish_count": 0,
    }


# ---------------------------------------------------------------------------
# Price structure
# ---------------------------------------------------------------------------

def compute_price_features(
    price: float,
    value_low: float | None = None,
    value_high: float | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Derive price-structure features relative to a value zone.

    If ``value_low``/``value_high`` are None, computes them dynamically
    from 52-week range in ``df``.

    Returns
    -------
    dict
        - ``in_buy_zone`` (bool)
        - ``discount_to_value`` (float)
        - ``value_low``, ``value_high``
        - ``pct_from_52w_low``, ``pct_from_52w_high``
    """
    if df is not None and len(df) >= 10:
        _52w_low = float(df["low"].min())
        _52w_high = float(df["high"].max())
        pct_range = _52w_high - _52w_low
        if value_low is None:
            value_low = _52w_low + pct_range * 0.2
        if value_high is None:
            value_high = _52w_low + pct_range * 0.4
        pct_from_low = (price - _52w_low) / (_52w_low + 1e-9) * 100
        pct_from_high = (price - _52w_high) / (_52w_high + 1e-9) * 100
    else:
        value_low = value_low or price * 0.95
        value_high = value_high or price * 1.05
        pct_from_low = 0.0
        pct_from_high = 0.0

    in_buy_zone = value_low <= price <= value_high
    discount_to_value = (value_high - price) / (value_high + 1e-9)

    return {
        "in_buy_zone": in_buy_zone,
        "discount_to_value": round(discount_to_value, 6),
        "value_low": round(value_low, 2),
        "value_high": round(value_high, 2),
        "pct_from_52w_low": round(pct_from_low, 2),
        "pct_from_52w_high": round(pct_from_high, 2),
    }


# ---------------------------------------------------------------------------
# Sentiment — tiếng Việt + tiếng Anh
# ---------------------------------------------------------------------------

_POSITIVE_VI = {
    "tăng", "tăng mạnh", "bứt phá", "hồi phục", "lạc quan", "kỳ vọng",
    "mua ròng", "tích lũy", "breakout", "vượt", "kết quả tốt",
    "doanh thu", "lợi nhuận", "tăng trưởng", "thặng dư",
}
_NEGATIVE_VI = {
    "giảm", "giảm mạnh", "bán ròng", "rủi ro", "lo ngại", "thận trọng",
    "áp lực", "chốt lời", "tháo chạy", "cảnh báo", "sụt giảm",
    "lỗ", "khó khăn", "bất ổn", "sụp đổ", "vi phạm",
}
_POSITIVE_EN = {
    "rally", "surge", "gain", "strong", "bullish", "positive",
    "growth", "beat", "up", "rise", "rises", "buy",
}
_NEGATIVE_EN = {
    "fall", "drop", "weak", "bearish", "negative", "loss", "miss",
    "down", "decline", "tensions", "weigh", "uncertainty", "sell",
}


def sentiment_score(headlines: list[str]) -> float:
    """
    Compute a normalised sentiment score in [-1, 1] from headlines.

    Supports both Vietnamese and English keywords.
    """
    if not headlines:
        return 0.0

    score = 0.0
    for headline in headlines:
        lower = headline.lower()
        words = set(lower.split())
        # English word-level
        score += len(words & _POSITIVE_EN)
        score -= len(words & _NEGATIVE_EN)
        # Vietnamese phrase-level
        for phrase in _POSITIVE_VI:
            if phrase in lower:
                score += 1
        for phrase in _NEGATIVE_VI:
            if phrase in lower:
                score -= 1

    # Normalise to [-1, 1]
    max_possible = len(headlines) * 5
    return round(max(-1.0, min(1.0, score / (max_possible + 1e-9))), 4)


def compute_sentiment(headlines: list[str]) -> dict[str, Any]:
    """
    Build a sentiment feature dict from raw news headlines.

    Returns
    -------
    dict
        - ``sentiment_score`` (float in [-1, 1])
        - ``sentiment_label`` (str): ``"POSITIVE"``, ``"NEGATIVE"``, ``"NEUTRAL"``
        - ``headline_count`` (int)
    """
    score = sentiment_score(headlines)

    if score > 0.05:
        label = "POSITIVE"
    elif score < -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "headline_count": len(headlines),
    }


# ---------------------------------------------------------------------------
# Smart Money Concepts (SMC) & ICT
# ---------------------------------------------------------------------------

def detect_swing_points(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Identify fractal swing highs and lows."""
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    
    for i in range(n, len(df) - n):
        # Swing High
        if all(df["high"].iloc[i] > df["high"].iloc[i-j] for j in range(1, n+1)) and \
           all(df["high"].iloc[i] > df["high"].iloc[i+j] for j in range(1, n+1)):
            df.at[df.index[i], "swing_high"] = True
            
        # Swing Low
        if all(df["low"].iloc[i] < df["low"].iloc[i-j] for j in range(1, n+1)) and \
           all(df["low"].iloc[i] < df["low"].iloc[i+j] for j in range(1, n+1)):
            df.at[df.index[i], "swing_low"] = True
            
    return df


def detect_fvg(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect Fair Value Gaps (FVG)."""
    fvgs = []
    if len(df) < 3:
        return fvgs
        
    for i in range(2, len(df)):
        # Bullish FVG (Gap between High of i-2 and Low of i)
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            fvgs.append({
                "type": "BULLISH",
                "top": float(df["low"].iloc[i]),
                "bottom": float(df["high"].iloc[i-2]),
                "index": i-1
            })
        # Bearish FVG (Gap between Low of i-2 and High of i)
        if df["high"].iloc[i] < df["low"].iloc[i-2]:
            fvgs.append({
                "type": "BEARISH",
                "top": float(df["low"].iloc[i-2]),
                "bottom": float(df["high"].iloc[i]),
                "index": i-1
            })
    return fvgs


def detect_structure(df: pd.DataFrame) -> dict[str, Any]:
    """Identify BOS (Break of Structure) and CHoCH (Change of Character)."""
    df_sw = detect_swing_points(df)
    highs = df_sw[df_sw["swing_high"]]
    lows = df_sw[df_sw["swing_low"]]
    
    if len(highs) < 2 or len(lows) < 2:
        return {"structure": "SIDEWAYS", "last_bos": None, "last_choch": None}
        
    last_high = highs["high"].iloc[-1]
    prev_high = highs["high"].iloc[-2]
    last_low = lows["low"].iloc[-1]
    prev_low = lows["low"].iloc[-2]
    
    current_close = df["close"].iloc[-1]
    
    structure_signal = "NEUTRAL"
    bos = None
    choch = None
    
    # Simple Logic for BOS/CHoCH
    if current_close > last_high:
        if last_high >= prev_high:
            bos = "BULLISH_BOS"
            structure_signal = "BULLISH"
        else:
            choch = "BULLISH_CHoCH"
            structure_signal = "BULLISH_REVERSAL"
            
    elif current_close < last_low:
        if last_low <= prev_low:
            bos = "BEARISH_BOS"
            structure_signal = "BEARISH"
        else:
            choch = "BEARISH_CHoCH"
            structure_signal = "BEARISH_REVERSAL"
            
    return {
        "structure": structure_signal,
        "bos": bos,
        "choch": choch,
        "last_high": float(last_high),
        "last_low": float(last_low)
    }


def compute_volatility_score(df: pd.DataFrame) -> float:
    """Compute volatility multiplier to trigger Scalping mode."""
    if len(df) < 20:
        return 1.0
    
    # Use Bollinger Bandwidth
    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    bandwidth = (4 * std) / (ma + 1e-9)
    
    current_bw = bandwidth.iloc[-1]
    avg_bw = bandwidth.rolling(50).mean().iloc[-1] if len(bandwidth) >= 50 else bandwidth.mean()
    
    score = current_bw / (avg_bw + 1e-9)
    return round(float(score), 2)


def compute_smc_features(df: pd.DataFrame) -> dict[str, Any]:
    """Orchestrate institutional SMC analysis."""
    if df is None or len(df) < 20:
        return {"error": "Insufficient data"}
        
    struct = detect_structure(df)
    fvgs = detect_fvg(df)
    vol_score = compute_volatility_score(df)
    
    # Institutional Mode Trigger
    mode = "SWING" if vol_score < 1.5 else "SCALPING"
    
    # Detect Liquidity Sweeps (Râu nến vượt qua đỉnh/đáy cũ rồi rút chân)
    latest_low = df["low"].iloc[-1]
    latest_high = df["high"].iloc[-1]
    latest_close = df["close"].iloc[-1]
    
    liquidity_sweep = None
    if latest_low < struct["last_low"] < latest_close:
        liquidity_sweep = "BULLISH_SWEEP"
    elif latest_high > struct["last_high"] > latest_close:
        liquidity_sweep = "BEARISH_SWEEP"
        
    return {
        "structure": struct["structure"],
        "bos": struct["bos"],
        "choch": struct["choch"],
        "fvgs": fvgs[-3:],  # Last 3 FVGs
        "volatility_score": vol_score,
        "trading_mode": mode,
        "liquidity_sweep": liquidity_sweep,
        "premium_zone": latest_close > (struct["last_high"] + struct["last_low"])/2
    }
