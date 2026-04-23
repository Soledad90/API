"""
Entry point for the VN Stock Market AI Agent.

Usage
-----
    # Single analysis (mặc định VCB)
    python main.py

    # Multi-ticker real-time (không dashboard)
    python main.py --symbols VCB,VIC,HPG,FPT --interval 30

    # Với dashboard (mở http://localhost:8000)
    python main.py --symbols VCB,VIC,HPG,FPT,MWG --dashboard

    # Ngoài giờ giao dịch (force run)
    python main.py --symbols VCB --no-trading-hours-check

Environment
-----------
Set OPENROUTER_API_KEY in a .env file (copy .env.example as a template).
Optional: LLM_MODEL (default: anthropic/claude-3.5-sonnet)
"""

import argparse
import logging
import sys
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VN Stock Market AI Agent — Real-Time Analysis"
    )
    parser.add_argument(
        "--symbols",
        default="VCB",
        help="Danh sách mã cổ phiếu, phân cách bằng dấu phẩy (vd: VCB,VIC,HPG). Mặc định: VCB",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Số giây giữa các lần scan (mặc định: 60). Không được < 15.",
    )
    parser.add_argument(
        "--portfolio",
        type=float,
        default=100_000_000.0,
        help="Giá trị danh mục VNĐ dùng cho tính toán position size (mặc định: 100,000,000)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Khởi chạy web dashboard tại http://localhost:8000",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port cho dashboard (mặc định: 8000)",
    )
    parser.add_argument(
        "--no-trading-hours-check",
        action="store_true",
        help="Bỏ qua kiểm tra giờ giao dịch (chạy 24/7, hữu ích khi test)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chỉ chạy 1 lần rồi thoát (không loop)",
    )
    return parser.parse_args()


def start_dashboard(port: int) -> None:
    """Start FastAPI dashboard server in a daemon thread."""
    try:
        import uvicorn
        from dashboard.server import app
        logger.info("🌐 Dashboard khởi động tại http://localhost:%d", port)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    except ImportError:
        logger.error("fastapi / uvicorn chưa được cài. Chạy: pip install fastapi uvicorn[standard]")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("Danh sách --symbols trống. Vui lòng cung cấp ít nhất 1 mã.")
        sys.exit(1)

    interval = max(15, args.interval)   # tối thiểu 15s để tránh rate-limit
    if interval != args.interval:
        logger.warning("interval < 15s không được phép — đặt thành 15s")

    logger.info("Symbols: %s | Interval: %ds | Portfolio: %s VNĐ", symbols, interval, f"{args.portfolio:,.0f}")

    broadcast_fn = None

    # ---- Dashboard ----
    if args.dashboard:
        from dashboard.server import _agent_state, get_broadcast_fn
        _agent_state["symbols"] = symbols
        _agent_state["running"] = True

        dash_thread = threading.Thread(
            target=start_dashboard, args=(args.port,), daemon=True
        )
        dash_thread.start()

        import time; time.sleep(1.5)   # dashboard'ın başlamasını bekle
        broadcast_fn = get_broadcast_fn()
        logger.info("Dashboard hazır: http://localhost:%d", args.port)

    # ---- Single run ----
    if args.once:
        from src.agent import run_agent_llm
        run_agent_llm(portfolio_value=args.portfolio, symbol=symbols[0])
        return

    # ---- Real-time loop ----
    from src.agent import run_realtime_loop
    run_realtime_loop(
        symbols=symbols,
        interval_seconds=interval,
        portfolio_value=args.portfolio,
        broadcast_fn=broadcast_fn,
        only_trading_hours=not args.no_trading_hours_check,
    )


if __name__ == "__main__":
    main()
