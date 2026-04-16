# AI Trading Agent (LLM-Driven)

An institutional-grade, LLM-driven trading agent built around a **Cognitive
Trading** architecture.  The system replaces brittle IF/ELSE rule trees with an
LLM reasoning core (OpenRouter / Claude) that synthesises market regime,
price-structure, and sentiment signals into a structured trading decision.

```
Market Data → Feature Engineering → LLM + Rule Engine (ensemble)
           → Risk Engine → Position Size → Memory / Feedback Loop
```

---

## Architecture

### Layers

| Layer | Module | Role |
|-------|--------|------|
| Data | `src/data.py` | Fetch market data & news (stub → replace with live feed) |
| Features | `src/features.py` | Market regime, price structure, sentiment scoring |
| LLM | `src/llm.py` | OpenRouter API call + strict JSON validation |
| Prompt | `src/prompt.py` | Hedge-fund-grade prompt builder |
| Risk | `src/risk.py` | Rule engine, ensemble combiner, position sizing |
| Memory | `src/memory.py` | Trade history, performance stats, feedback loop |
| Agent | `src/agent.py` | Multi-agent orchestrator (`run_agent_llm`) |

### Multi-Agent sub-system (`src/agent.py`)

```
MacroAgent     → market regime features
TechnicalAgent → price-structure features
SentimentAgent → news sentiment features
DecisionAgent  → LLM reasoning core + rule engine ensemble
```

### Ensemble model

```
Final Decision = 50% Rule Engine + 50% LLM
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Soledad90/API.git
cd API
pip install -r requirements.txt
```

### 2. Configure the API key

```bash
cp .env.example .env
# Open .env and set your OPENROUTER_API_KEY
```

> **Security note**: never commit `.env` to version control.
> The key is read exclusively from the environment variable
> `OPENROUTER_API_KEY`.

### 3. Run the agent

```bash
python main.py
```

Sample output:

```
🧠 AI DECISION:
{
  "decision": "WATCH",
  "confidence": 0.62,
  "reasoning": "Ensemble (rule=WATCH, llm=WATCH, score=0.00). ...",
  "risk_level": "MEDIUM",
  "position_size_usd": 0.0
}

📊 PERFORMANCE STATS:
{
  "total_trades": 1,
  "wins": 0,
  "losses": 0,
  "win_rate": 0.0,
  "avg_confidence": 0.62
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 62 unit tests cover feature engineering, LLM output validation, risk
engine, ensemble combiner, memory / feedback loop, and prompt building.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM non-determinism | `temperature=0.1` in all LLM calls |
| LLM hallucination | Strict JSON schema validation with `parse_llm_json` |
| Runaway losses | Feedback loop reduces confidence after recent losses |
| No PnL optimisation | Backtest before going live; add reinforcement loop |

---

## Roadmap

- [ ] Replace data stubs with live broker / exchange API
- [ ] Integrate FinBERT for production-quality sentiment scoring
- [ ] Add backtesting harness
- [ ] Reinforcement-learning feedback loop (update on closed trade P&L)
- [ ] Macro / Technical / Sentiment agents calling specialised LLM models
