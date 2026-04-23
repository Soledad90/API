"""
Main LLM-driven trading agent — VN Stock Market Real-Time Edition.

Architecture (Cognitive Trading):

    Market Data → Feature Engineering → LLM + Rule Engine → Ensemble
                                      → Risk Engine → Position Size
                                      → Memory / Feedback Loop

Multi-agent sub-system:
  - MacroAgent    : market regime (breadth)
  - TechnicalAgent: price structure + TA indicators
  - SentimentAgent: news / sentiment (tiếng Việt & EN)
  - DecisionAgent : LLM reasoning core + rule engine ensemble

Real-time loop:
  - Scans multiple VN tickers every N seconds
  - Auto-pauses outside VN trading hours (09:00–11:30, 13:00–14:45 ICT)
  - Broadcasts results to WebSocket clients when dashboard is enabled
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any

from .data import (
    fetch_news,
    get_market_data,
    is_trading_hours,
)
from .features import (
    compute_market_features,
    compute_price_features,
    compute_sentiment,
    compute_technical_indicators,
    compute_smc_features,
)
from .utils import resample_ohlcv
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
    """Evaluates price structure and technical indicators."""

    def analyse(
        self,
        price: float,
        df=None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (price_feat, tech_feat)."""
        price_feat = compute_price_features(price, df=df)
        tech_feat = compute_technical_indicators(df) if df is not None else {}
        logger.debug("TechnicalAgent price_feat: %s", price_feat)
        logger.debug("TechnicalAgent tech_feat trend=%s RSI=%s", tech_feat.get("trend"), tech_feat.get("rsi"))
        return price_feat, tech_feat


class SentimentAgent:
    """Evaluates news sentiment (tiếng Việt + English)."""

    def analyse(self, headlines: list[str]) -> dict[str, Any]:
        features = compute_sentiment(headlines)
        logger.debug("SentimentAgent features: %s", features)
        return features


class DecisionAgent:
    """
    Synthesises macro, technical, and sentiment signals.

    Uses an LLM as the primary reasoning core (50% weight) combined with
    a deterministic rule engine (50% weight) to produce a blended decision.
    """

    def decide(
        self,
        market_feat: dict[str, Any],
        price_feat: dict[str, Any],
        sentiment_feat: dict[str, Any],
        tech_feat: dict[str, Any],
        price: float,
        symbol: str,
        headlines: list[str],
        smc_feat: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (llm_result, rule_result)."""
        # Rule engine (deterministic, fast)
        rule_result = rule_engine_decision(
            market_feat, price_feat, sentiment_feat, tech_feat, smc_feat
        )

        # LLM reasoning core
        prompt = build_prompt(
            market_feat, price_feat, sentiment_feat, price,
            symbol=symbol, tech_feat=tech_feat, headlines=headlines,
            smc_feat=smc_feat,
        )
        try:
            raw_llm = call_llm(prompt)
            llm_result = parse_llm_json(raw_llm)
        except Exception as exc:
            logger.error("LLM decision failed: %s — falling back to rules", exc)
            llm_result = rule_result.copy()
            llm_result["reasoning"] = f"[LLM fallback] {rule_result['reasoning']}"

        return llm_result, rule_result


# ---------------------------------------------------------------------------
# Single-symbol analysis pipeline
# ---------------------------------------------------------------------------

def analyse_symbol(
    symbol: str = "VCB",
    portfolio_value: float = 100_000_000.0,
    persist_memory: bool = True,
    headlines: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the full analysis pipeline for a single symbol.

    Parameters
    ----------
    symbol : str
        VN stock ticker (e.g. ``"VCB"``, ``"VIC"``, ``"HPG"``).
    portfolio_value : float
        Total portfolio value in VND for position sizing.
    persist_memory : bool
        Whether to write the trade record to the memory file.
    headlines : list[str], optional
        Pre-fetched news headlines (avoids repeated RSS calls in batch mode).

    Returns
    -------
    dict
        Full analysis result with decision, performance, and all features.
    """
    timestamp = datetime.now().isoformat()

    # ---- 1. Data (Multi-TF) ----
    from .data import get_market_history, get_market_data
    
    market_data = get_market_data(symbol)   # get 1D data
    df_1h = get_market_history(symbol, period=200, interval="1H")
    
    if df_1h.empty:
        df_1h = market_data.get("df")  # fallback to 1D if 1H missing
        
    df_4h = resample_ohlcv(df_1h, "4h")
    
    if headlines is None:
        headlines = fetch_news()

    # ---- 2. Specialist agents ----
    macro_agent    = MacroAgent()
    technical_agent = TechnicalAgent()
    sentiment_agent = SentimentAgent()

    market_feat            = macro_agent.analyse(market_data)
    price_feat, tech_feat  = technical_agent.analyse(
        market_data["price"], df=market_data.get("df")
    )
    sentiment_feat         = sentiment_agent.analyse(headlines)
    
    # SMC Analysis on LTF (1H)
    smc_feat = compute_smc_features(df_1h)

    # ---- 3 & 4. Decision + ensemble ----
    decision_agent = DecisionAgent()
    llm_result, rule_result = decision_agent.decide(
        market_feat, price_feat, sentiment_feat, tech_feat,
        market_data["price"], symbol, headlines,
        smc_feat=smc_feat,
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
    final["position_size_vnd"] = position_size

    # ---- 7. Memory ----
    trade_record = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": market_data["price"],
        "market_feat": market_feat,
        "price_feat": price_feat,
        "tech_feat": {k: v for k, v in tech_feat.items() if v is not None},
        "smc_feat": smc_feat,
        "sentiment_feat": sentiment_feat,
        "rule_result": rule_result,
        "llm_result": llm_result,
        "final_decision": final,
        "outcome": None,
        "confidence": final["confidence"],
    }
    if persist_memory:
        save_memory(trade_record)

    stats = compute_performance_stats(history)

    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": market_data["price"],
        "decision": final,
        "tech": tech_feat,
        "market": market_feat,
        "sentiment": sentiment_feat,
        "smc": smc_feat,
        "performance": stats,
        "trading_mode": smc_feat.get("trading_mode", "SWING"),
        "sector": __import__("src.data", fromlist=["TICKER_SECTORS"]).TICKER_SECTORS.get(symbol, "Other"),
        "data_source": market_data.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Legacy single-run entry point (backwards compat)
# ---------------------------------------------------------------------------

def run_agent_llm(
    portfolio_value: float = 100_000_000.0,
    persist_memory: bool = True,
    symbol: str = "VCB",
) -> dict[str, Any]:
    """
    Run the full LLM-driven trading agent pipeline (single run).

    Parameters
    ----------
    portfolio_value : float
        Total portfolio value in VND.
    persist_memory : bool
        Whether to write the trade record to the memory file.
    symbol : str
        Default symbol to analyse.

    Returns
    -------
    dict
        Final blended decision, position size, and performance stats.
    """
    result = analyse_symbol(symbol, portfolio_value, persist_memory)
    final  = result["decision"]
    stats  = result["performance"]

    print(f"\n[{result['symbol']}] Gia: {result['price']:,.0f} VND | Nguon: {result['data_source']}")
    print("\nAI DECISION:")
    print(json.dumps(final, indent=2, ensure_ascii=True))
    print("\nPERFORMANCE STATS:")
    print(json.dumps(stats, indent=2, ensure_ascii=True))

    return result


# ---------------------------------------------------------------------------
# Real-time multi-ticker loop
# ---------------------------------------------------------------------------

def run_realtime_loop(
    symbols: list[str] | None = None,
    interval_seconds: int = 30,
    portfolio_value: float = 100_000_000.0,
    broadcast_fn=None,
    only_trading_hours: bool = True,
) -> None:
    """
    Continuously scan multiple VN tickers and print/broadcast decisions.

    Parameters
    ----------
    symbols : list[str]
        List of VN ticker codes. Default: [VCB, VIC, HPG, FPT, MWG].
    interval_seconds : int
        Seconds between scans (default 30).
    portfolio_value : float
        Portfolio value in VND for position sizing.
    broadcast_fn : callable, optional
        Async function ``(result_dict) → None`` to send results to dashboard.
    only_trading_hours : bool
        If True, pause outside VN trading hours. Default True.
    """
    if symbols is None:
        symbols = ["VCB", "VIC", "VHM", "VRE", "HPG", "FPT", "MWG", "VNM", "LCG", "ACB", "SSI", "CTG", "CII", "MSN", "MBB", "TCB"]

    logger.info("Real-time loop started | Symbols: %s | Interval: %ds", symbols, interval_seconds)
    print(f"\n{'='*60}")
    print(f"  VN STOCK AI AGENT --- REAL-TIME MODE")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Interval: {interval_seconds}s | Portfolio: {portfolio_value:,.0f} VND")
    print(f"{'='*60}\n")

    while True:
        now = datetime.now()

        if only_trading_hours and not is_trading_hours(now):
            next_check = 60   # kiểm tra lại sau 60s
            print(f"[PAUSE] {now.strftime('%H:%M:%S')} - Ngoai gio giao dich. Cho {next_check}s...")
            time.sleep(next_check)
            continue

        # Fetch news một lần cho cả batch
        try:
            headlines = fetch_news()
        except Exception as exc:
            logger.warning("fetch_news lỗi: %s", exc)
            logger.warning("fetch_news loi: %s", exc)
            headlines = []

        scan_results = []
        for symbol in symbols:
            try:
                print(f"\n[>>] Phan tich {symbol} [{now.strftime('%H:%M:%S')}]...")
                result = analyse_symbol(
                    symbol, portfolio_value,
                    persist_memory=True,
                    headlines=headlines,
                )
                scan_results.append(result)

                d = result["decision"]
                tech = result.get("tech", {})
                emoji = {"BUY": "[BUY]", "SELL": "[SELL]", "WATCH": "[WATCH]", "NO_TRADE": "[--]"}  .get(
                    d["decision"], "[?]")
                print(
                    f"  {emoji} {symbol}: {d['decision']} | "
                    f"Confidence: {d['confidence']:.0%} | "
                    f"Risk: {d['risk_level']} | "
                    f"RSI: {tech.get('rsi', 'N/A')} | "
                    f"Gia: {result['price']:,.0f}"
                )
                if d["decision"] in ("BUY", "SELL"):
                    print(f"     $ Vi the de xuat: {d.get('position_size_vnd', 0):+,.0f} VND")
                    reasoning_ascii = d['reasoning'][:100].encode('ascii', errors='replace').decode('ascii')
                    print(f"     > {reasoning_ascii}...")

                # Broadcast to dashboard
                if broadcast_fn is not None:
                    try:
                        asyncio.get_event_loop().run_until_complete(
                            broadcast_fn(result)
                        )
                    except Exception as exc:
                        logger.debug("Broadcast error: %s", exc)

            except Exception as exc:
                logger.error("Lỗi phân tích %s: %s", symbol, exc)
            
            # Thêm độ trễ nhỏ để tránh Rate Limit của vnstock (Guest: 20 req/min)
            time.sleep(3)

            # ---- Sector Flow Analysis ----
            sector_data = {}
            for r in scan_results:
                sec = r.get("sector", "Other")
                if sec not in sector_data:
                    sector_data[sec] = {"change": [], "vol": [], "count": 0}
                
                change = r.get("tech", {}).get("price_change_pct", 0)
                vol = r.get("tech", {}).get("vol_ratio", 1.0)
                
                sector_data[sec]["change"].append(change)
                sector_data[sec]["vol"].append(vol)
                sector_data[sec]["count"] += 1
            
            final_sectors = {}
            for sec, vals in sector_data.items():
                avg_change = sum(vals["change"]) / len(vals["change"])
                avg_vol = sum(vals["vol"]) / len(vals["vol"])
                
                # Flow label: Cường độ dòng tiền
                if avg_change > 0.5 and avg_vol > 1.2: flow = "LEADING (Hut tien)"
                elif avg_change > -0.2 and avg_vol > 1.0: flow = "IMPROVING (Tich luy)"
                elif avg_change < -0.5: flow = "LAGGING (Suy yeu)"
                else: flow = "NEUTRAL"
                
                final_sectors[sec] = {
                    "avg_change": round(avg_change, 2),
                    "avg_vol": round(avg_vol, 2),
                    "count": vals["count"],
                    "flow": flow
                }
            
            # Broadcast sector update
            if broadcast_fn is not None:
                try:
                    asyncio.get_event_loop().run_until_complete(
                        broadcast_fn({"type": "sector_update", "data": final_sectors})
                    )
                except Exception: pass

            print(f"\nScan hoan tat. Cho {interval_seconds}s...\n{'---'*17}")
            time.sleep(interval_seconds)
