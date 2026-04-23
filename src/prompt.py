"""
Hedge-fund-grade prompt builder — VN Stock Market.

Builds a rich context prompt that includes:
  - Thông tin mã cổ phiếu & bối cảnh thị trường VN
  - Technical indicators (MA, RSI, MACD, BB, Volume)
  - Tâm lý thị trường (sentiment + headlines)
  - Yêu cầu trả lời JSON strict
"""

from __future__ import annotations

from typing import Any


_JSON_SCHEMA = """{
  "decision": "BUY" | "SELL" | "WATCH" | "NO_TRADE",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<Giải thích bằng tiếng Việt, tập trung vào cấu trúc thị trường, thanh khoản và xác nhận>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "trading_mode_detected": "SWING" | "SCALPING",
  "key_signals": ["<signal_1>", "<signal_2>", ...],
  "position_size_vnd": <float (số tiền VNĐ nên vào lệnh nếu BUY, 0 nếu khác)>
}"""


def build_prompt(
    market_feat: dict[str, Any],
    price_feat: dict[str, Any],
    sentiment_feat: dict[str, Any],
    price: float,
    symbol: str = "UNKNOWN",
    tech_feat: dict[str, Any] | None = None,
    headlines: list[str] | None = None,
    smc_feat: dict[str, Any] | None = None,
) -> str:
    """
    Build a structured analysis prompt for the LLM based on Institutional SMC/ICT.
    """
    lines: list[str] = [
        f"## [Institutional Analysis] {symbol} — Vietnam Stock Market (HOSE/HNX)",
        "",
        "Bạn là một Chuyên gia Phân tích Hệ thống và Chiến lược cấp cao tại một Quỹ đầu cơ.",
        "Nhiệm vụ: Phân tích dựa trên Dòng tiền thông minh (SMC), Cấu trúc thị trường và Thanh khoản.",
        "",
        "### 1. Market Context & Data",
        f"- Sàn niêm yết: HOSE/HNX",
        f"- Giá hiện tại: {price:,.0f} VND",
        f"- Market Breadth: {market_feat.get('market_regime', 'N/A')} (Ratio: {market_feat.get('breadth_ratio', 'N/A')})",
        "",
    ]

    if smc_feat:
        mode = smc_feat.get("trading_mode", "SWING")
        lines += [
            "### 2. Smart Money Concepts (SMC) & ICT",
            f"- **CHẾ ĐỘ GIAO DỊCH:** **{mode}** (Dựa trên Volatility Score: {smc_feat.get('volatility_score', 1.0)})",
            f"- Cấu trúc thị trường: **{smc_feat.get('structure', 'NEUTRAL')}**",
            f"- Break of Structure (BOS): {smc_feat.get('bos') or 'None'}",
            f"- Change of Character (CHoCH): {smc_feat.get('choch') or 'None'}",
            f"- Liquidity Sweep: **{smc_feat.get('liquidity_sweep') or 'No current sweep detected'}**",
            f"- Premium/Discount: {'PREMIUM (Sell side)' if smc_feat.get('premium_zone') else 'DISCOUNT (Buy side)'}",
            f"- Fair Value Gaps (FVG): {len(smc_feat.get('fvgs', []))} gaps gần nhất detected.",
            "",
        ]

    if tech_feat:
        lines += [
            "### 3. Confluence (Technical Indicators)",
            f"- Trend: {tech_feat.get('trend', 'N/A')} (MA5/20/50: {tech_feat.get('ma_signal', 'N/A')})",
            f"- Momentum: RSI={tech_feat.get('rsi', 'N/A')} | MACD={tech_feat.get('macd_cross', 'N/A')}",
            f"- Volume Profile: {tech_feat.get('vol_signal', 'N/A')} (Ratio: {tech_feat.get('vol_ratio', 'N/A')}x)",
            "",
        ]

    lines += [
        "### 4. Market Sentiment & Sentiment Analysis",
        f"- Score: {sentiment_feat.get('sentiment_score', 0):.4f} ({sentiment_feat.get('sentiment_label', 'N/A')})",
    ]

    if headlines:
        lines += ["- Headlines gần đây:"]
        for h in headlines[:5]:
            lines.append(f"  • {h}")
    lines.append("")

    lines += [
        "### 5. Institutional Instruction",
        "Hãy thực hiện phân tích 'Wait -> Watch -> Confirm -> Execute':",
        "1. XÁC ĐỊNH BIAS: Dựa trên bối cảnh thị trường và cấu trúc HTF (D1/4H).",
        f"2. NẾU LÀ {mode}: Tập trung vào các nhịp hồi về POI (OB/FVG) hoặc quét thanh khoản.",
        "3. ĐIỂM VÀO LỆNH (ENTRY): Ưu tiên tại các vùng DISCOUNT nếu BUY và PREMIUM nếu SELL.",
        "4. QUẢN TRỊ RỦI RO: Đề xuất vị thế VNĐ hợp lý dựa trên 1-2% tài khoản.",
        "",
        "TRẢ LỜI ĐÚNG JSON SCHEMA SAU (KHÔNG THÊM GIẢI THÍCH NGOÀI):",
        "",
        _JSON_SCHEMA,
    ]

    return "\n".join(lines)
