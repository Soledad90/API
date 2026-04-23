"""
Tests for the data fetching stubs.
"""

from src.data import get_market_data, fetch_news


class TestGetMarketData:
    def test_returns_dict(self):
        data = get_market_data()
        assert isinstance(data, dict)

    def test_has_required_keys(self):
        data = get_market_data()
        assert "price" in data
        assert "advance" in data
        assert "decline" in data

    def test_price_is_float(self):
        data = get_market_data()
        assert isinstance(data["price"], float)


class TestFetchNews:
    def test_returns_list(self):
        news = fetch_news()
        assert isinstance(news, list)

    def test_contains_strings(self):
        news = fetch_news()
        assert all(isinstance(h, str) for h in news)
