"""
Market data and news fetching — Real-Time VN Stock Market.

Data sources:
- vnstock3  : Live price, OHLCV, breadth data from HOSE / HNX / SSI
- yfinance  : VN-Index ETF & international reference (E1VFVN30)
- CafeF RSS : Vietnamese financial news
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any

import feedparser
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAFEF_RSS = "https://cafef.vn/thi-truong-chung-khoan.rss"
_VNEXPRESS_RSS = "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss"

_TRADING_HOURS = [
    ("09:00", "11:30"),   # Phiên sáng
    ("13:00", "14:45"),   # Phiên chiều (ATC kết thúc 14:45)
]

# Mapping Ticker -> Sector cho mục đích phân tích dòng tiền
TICKER_SECTORS = {
    "VCB": "Banking", "ACB": "Banking", "CTG": "Banking", "MBB": "Banking", "TCB": "Banking", "STB": "Banking",
    "VIC": "Real Estate", "VHM": "Real Estate", "VRE": "Real Estate", "LCG": "Real Estate", "CII": "Real Estate", "NVL": "Real Estate", "DXG": "Real Estate",
    "HPG": "Steel", "HSG": "Steel", "NKG": "Steel",
    "FPT": "Technology", "CTR": "Technology",
    "MWG": "Retail", "VNM": "Consumer", "MSN": "Consumer", "PNJ": "Retail",
    "SSI": "Securities", "VND": "Securities", "VCI": "Securities", "HCM": "Securities"
}


# ---------------------------------------------------------------------------
# Trading session helpers
# ---------------------------------------------------------------------------

def is_trading_hours(now: datetime | None = None) -> bool:
    """Return True if ``now`` is within VN stock exchange trading hours."""
    now = now or datetime.now()
    if now.weekday() >= 5:          # Thứ 7, CN
        return False
    t = now.strftime("%H:%M")
    for start, end in _TRADING_HOURS:
        if start <= t <= end:
            return True
    return False


def is_trading_day(dt: date | None = None) -> bool:
    """Return True if ``dt`` is a weekday (Mon–Fri)."""
    dt = dt or date.today()
    return dt.weekday() < 5


# ---------------------------------------------------------------------------
# Live market data — vnstock3
# ---------------------------------------------------------------------------

def get_market_history(symbol: str = "VCB", period: int = 60, interval: str = "1D") -> pd.DataFrame:
    """Generic helper to fetch historical OHLCV data."""
    try:
        try:
            from vnstock import Vnstock as _VnstockCls
        except ImportError:
            from vnstock3 import Vnstock as _VnstockCls

        _SOURCES = ["KBS", "VCI", "FMP"]
        stock = None
        for _src in _SOURCES:
            try:
                stock = _VnstockCls().stock(symbol=symbol, source=_src)
                break
            except Exception:
                continue
        if stock is None:
            raise RuntimeError(f"vnstock source error for {symbol}")

        import datetime as _dt
        # Nếu interval là 1H/1m ta cần lấy nhiều ngày hơn để đủ period nến
        days_back = period * 1.5 if interval == "1D" else (period / 5 + 2)
        start_dt = _dt.date.today() - _dt.timedelta(days=int(days_back))
        start = start_dt.strftime("%Y-%m-%d")
        end = date.today().strftime("%Y-%m-%d")

        df: pd.DataFrame = stock.quote.history(start=start, end=end, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)
            df.index = pd.to_datetime(df["time"])
        else:
            df = df.sort_index()

        # Giá đơn vị nghìn VND -> VND
        for col in ["open", "high", "low", "close"]:
            if df[col].max() < 10_000:
                df[col] = df[col] * 1000
        return df
    except Exception as e:
        logger.error("get_market_history error %s: %s", symbol, e)
        return pd.DataFrame()


def get_market_data(symbol: str = "VCB", period: int = 60) -> dict[str, Any]:
    """
    Fetch live OHLCV data and breadth.
    """
    df = get_market_history(symbol, period, interval="1D")
    if df.empty:
        # Fallback to yfinance if vnstock fails
        try:
            import yfinance as yf
            ticker = f"{symbol}.VN" if len(symbol) == 3 else symbol
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)
            df.columns = [c.lower() for c in df.columns]
        except Exception as e:
            logger.error("yfinance fallback failed for %s: %s", symbol, e)
            raise ValueError(f"Không thể lấy dữ liệu cho {symbol}")

    if df.empty:
         raise ValueError(f"Không thể lấy dữ liệu cho {symbol}")

    latest = df.iloc[-1]
    price = float(latest["close"])
    advance, decline = _estimate_breadth(df)

    return {
        "symbol": symbol,
        "price": price,
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "volume": float(latest["volume"]),
        "advance": advance,
        "decline": decline,
        "df": df,
        "source": "vnstock/yf"
    }


def _get_market_data_yfinance(symbol: str, period: int = 60) -> dict[str, Any]:
    """Fallback: lấy dữ liệu từ yfinance (chủ yếu dùng cho test offline)."""
    try:
        import yfinance as yf
        # Map VN tickers sang Yahoo Finance format
        yf_map = {
            "VNI": "^VNINDEX",
            "VFVN30": "E1VFVN30.BK",
        }
        yf_symbol = yf_map.get(symbol, f"{symbol}.VN")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{period}d")

        if df.empty:
            logger.warning("yfinance cũng empty cho %s — dùng stub", symbol)
            return _stub_market_data(symbol)

        df.columns = [c.lower() for c in df.columns]
        latest = df.iloc[-1]
        advance, decline = _estimate_breadth(df)

        return {
            "symbol": symbol,
            "price": float(latest["close"]),
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "volume": float(latest["volume"]),
            "advance": advance,
            "decline": decline,
            "df": df,
            "source": "yfinance",
        }
    except Exception as exc:
        logger.error("yfinance lỗi: %s — dùng stub", exc)
        return _stub_market_data(symbol)


def _estimate_breadth(df: pd.DataFrame) -> tuple[int, int]:
    """
    Ước tính advance/decline từ momentum của price series.
    Đây là proxy đơn giản; thay bằng API breadth thật trong production.
    """
    if len(df) < 2:
        return 200, 200
    change = df["close"].iloc[-1] - df["close"].iloc[-2]
    if change > 0:
        return 350, 150
    elif change < 0:
        return 150, 350
    return 250, 250


def _stub_market_data(symbol: str) -> dict[str, Any]:
    """Stub tối giản khi cả vnstock3 lẫn yfinance đều fail."""
    logger.warning("Dùng stub data cho %s", symbol)
    import numpy as np
    n = 60
    prices = 50_000 + np.random.randn(n).cumsum() * 500
    df = pd.DataFrame({
        "open": prices * 0.999,
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    })
    return {
        "symbol": symbol,
        "price": float(prices[-1]),
        "open": float(prices[-1] * 0.999),
        "high": float(prices[-1] * 1.005),
        "low": float(prices[-1] * 0.995),
        "volume": 2_000_000.0,
        "advance": 200,
        "decline": 300,
        "df": df,
        "source": "stub",
    }


# ---------------------------------------------------------------------------
# Market breadth — HOSE toàn thị trường
# ---------------------------------------------------------------------------

def get_market_breadth() -> dict[str, int]:
    """
    Lấy số mã tăng/giảm toàn sàn HOSE.

    Returns
    -------
    dict
        ``advance`` và ``decline`` counts.
    """
    try:
        from vnstock3 import Vnstock
        vs = Vnstock()
        # Lấy toàn bộ cổ phiếu HOSE
        listing = vs.stock(source="VCI").listing.all_symbols()
        hose = listing[listing["exchange"].str.upper() == "HOSE"]["ticker"].tolist()

        advance = decline = 0
        # Lấy giá intraday cho tất cả mã (batch)
        for ticker in hose[:50]:   # giới hạn 50 mã để tránh rate-limit
            try:
                data = get_market_data(ticker, period=5)
                df = data["df"]
                if len(df) >= 2:
                    chg = df["close"].iloc[-1] - df["close"].iloc[-2]
                    if chg > 0:
                        advance += 1
                    elif chg < 0:
                        decline += 1
            except Exception:
                pass
        return {"advance": max(advance, 1), "decline": max(decline, 1)}
    except Exception as exc:
        logger.warning("get_market_breadth lỗi: %s — dùng ước tính", exc)
        return {"advance": 200, "decline": 300}


# ---------------------------------------------------------------------------
# News — CafeF + VnExpress RSS
# ---------------------------------------------------------------------------

def fetch_news(max_items: int = 10) -> list[str]:
    """
    Lấy tiêu đề tin tức tài chính mới nhất từ CafeF & VnExpress RSS.

    Parameters
    ----------
    max_items : int
        Số lượng tối đa headlines trả về.

    Returns
    -------
    list[str]
        Danh sách headline strings (tiếng Việt có dấu).
    """
    headlines: list[str] = []
    feeds = [_CAFEF_RSS, _VNEXPRESS_RSS]

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[: max_items // 2]:
                title = entry.get("title", "").strip()
                if title:
                    headlines.append(title)
        except Exception as exc:
            logger.warning("RSS lỗi %s: %s", url, exc)

    if not headlines:
        logger.warning("Không lấy được RSS — dùng stub headlines")
        headlines = [
            "Thị trường chứng khoán VN mở cửa thận trọng sau kỳ nghỉ lễ.",
            "Khối ngoại mua ròng mạnh nhóm cổ phiếu ngân hàng.",
            "VN-Index điều chỉnh nhẹ trước áp lực chốt lời cuối phiên.",
        ]

    return headlines[:max_items]
