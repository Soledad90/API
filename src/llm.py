"""
OpenRouter LLM integration — VN Stock Market agent.

The API key is read from the OPENROUTER_API_KEY environment variable.
Never hard-code secrets in source files.
"""

import json
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-3.5-sonnet")

_VALID_DECISIONS = {"BUY", "SELL", "NO_TRADE", "WATCH"}
_VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


def call_llm(
    prompt: str,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> str:
    """
    Send a prompt to the OpenRouter API and return the raw text response.

    Parameters
    ----------
    prompt : str
        The user prompt to send.
    temperature : float
        Sampling temperature (lower = more deterministic). Default 0.1.
    max_retries : int
        Number of retries with exponential back-off on transient errors.

    Returns
    -------
    str
        The assistant's response text.

    Raises
    ------
    EnvironmentError
        If OPENROUTER_API_KEY is not set.
    RuntimeError
        If all retries are exhausted.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY chưa được thiết lập. "
            "Copy .env.example → .env và điền key của bạn."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Soledad90/API",
        "X-Title": "VN Stock Market AI Agent",
    }

    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Bạn là chuyên gia phân tích chứng khoán Việt Nam tại một quỹ đầu tư "
                    "tổ chức hàng đầu. Bạn phân tích dữ liệu kỹ thuật và cơ bản để đưa ra "
                    "quyết định giao dịch chuyên nghiệp. Luôn trả lời bằng JSON hợp lệ."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=45
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug("LLM response (attempt %d): %s", attempt, content[:200])
            return content
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("LLM timeout (attempt %d/%d) — retry in %ds", attempt, max_retries, wait)
            time.sleep(wait)
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code in (429, 503):
                wait = 2 ** attempt
                logger.warning("LLM rate-limit/503 — retry in %ds", wait)
                time.sleep(wait)
            else:
                raise
        except Exception as exc:
            last_exc = exc
            logger.error("LLM unexpected error: %s", exc)
            break

    raise RuntimeError(
        f"LLM call thất bại sau {max_retries} lần thử: {last_exc}"
    )


def parse_llm_json(raw: str) -> dict[str, Any]:
    """
    Extract and validate the JSON object embedded in *raw* LLM output.

    Expected keys: ``decision``, ``confidence``, ``reasoning``, ``risk_level``.
    Valid decisions: BUY, SELL, NO_TRADE, WATCH.

    Parameters
    ----------
    raw : str
        Raw string returned by the LLM.

    Returns
    -------
    dict
        Parsed and validated decision dictionary.

    Raises
    ------
    ValueError
        If parsing fails or required keys are missing / invalid.
    """
    text = raw.strip()

    # Strip optional markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)

    # Find first '{' in case there's text before the JSON
    start = text.find("{")
    if start > 0:
        text = text[start:]
    end = text.rfind("}")
    if end != -1 and end < len(text) - 1:
        text = text[: end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM không trả về JSON hợp lệ: {exc}\nRaw:\n{raw}"
        ) from exc

    required = {"decision", "confidence", "reasoning", "risk_level"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"LLM JSON thiếu keys: {missing}")

    # Normalise decision to uppercase
    data["decision"] = str(data["decision"]).upper().strip()
    if data["decision"] not in _VALID_DECISIONS:
        raise ValueError(
            f"decision không hợp lệ: '{data['decision']}'. "
            f"Cần một trong {_VALID_DECISIONS}."
        )

    data["risk_level"] = str(data["risk_level"]).upper().strip()
    if data["risk_level"] not in _VALID_RISK_LEVELS:
        raise ValueError(
            f"risk_level không hợp lệ: '{data['risk_level']}'. "
            f"Cần một trong {_VALID_RISK_LEVELS}."
        )

    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence phải trong [0, 1], nhận được {confidence}."
        )
    data["confidence"] = confidence

    return data
