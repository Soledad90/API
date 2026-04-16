"""
Market data and news fetching.

In production these functions connect to a live data feed or broker API.
The stubs here return sample data so the rest of the pipeline can run
without external dependencies.
"""

from __future__ import annotations

from typing import Any


def get_market_data() -> dict[str, Any]:
    """
    Return current market data.

    Returns
    -------
    dict
        Keys:
        - ``price`` (float): current index / asset price.
        - ``advance`` (int): number of advancing issues.
        - ``decline`` (int): number of declining issues.

    Notes
    -----
    Replace the stub body with a real broker / data-feed call in
    production, e.g.::

        import yfinance as yf
        ticker = yf.Ticker("^GSPC")
        price = ticker.fast_info["lastPrice"]
    """
    return {
        "price": 17_500.0,
        "advance": 200,
        "decline": 300,
    }


def fetch_news() -> list[str]:
    """
    Return a list of recent news headlines relevant to the traded asset.

    Returns
    -------
    list[str]
        Recent headline strings.

    Notes
    -----
    Replace the stub body with a live news API call in production, e.g.::

        import requests
        resp = requests.get("https://newsapi.org/v2/top-headlines", ...)
        return [a["title"] for a in resp.json()["articles"]]
    """
    return [
        "Markets open cautiously amid Fed uncertainty.",
        "Tech stocks rally on strong earnings reports.",
        "Geopolitical tensions weigh on energy prices.",
    ]
