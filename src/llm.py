"""
OpenRouter LLM integration.

The API key is read from the OPENROUTER_API_KEY environment variable.
Never hard-code secrets in source files.
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """
    Send a prompt to the OpenRouter API and return the raw text response.

    Parameters
    ----------
    prompt : str
        The user prompt to send.
    temperature : float
        Sampling temperature (lower = more deterministic).  Default 0.1 to
        minimise non-determinism and hallucination risk.

    Returns
    -------
    str
        The assistant's response text.

    Raises
    ------
    EnvironmentError
        If OPENROUTER_API_KEY is not set.
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional hedge fund trader.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def parse_llm_json(raw: str) -> dict:
    """
    Extract and validate the JSON object embedded in *raw* LLM output.

    The LLM is instructed to return strict JSON, but it may wrap it in
    markdown fences.  This helper strips fences and parses the JSON,
    then validates the required keys.

    Parameters
    ----------
    raw : str
        Raw string returned by the LLM.

    Returns
    -------
    dict
        Parsed and validated decision dictionary with keys:
        ``decision``, ``confidence``, ``reasoning``, ``risk_level``.

    Raises
    ------
    ValueError
        If parsing fails or required keys are missing.
    """
    text = raw.strip()

    # Strip optional markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON: {exc}\nRaw output:\n{raw}") from exc

    required = {"decision", "confidence", "reasoning", "risk_level"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"LLM JSON missing required keys: {missing}")

    valid_decisions = {"BUY", "NO_TRADE", "WATCH"}
    if data["decision"] not in valid_decisions:
        raise ValueError(
            f"Invalid decision value '{data['decision']}'. "
            f"Expected one of {valid_decisions}."
        )

    valid_risk_levels = {"LOW", "MEDIUM", "HIGH"}
    if data["risk_level"] not in valid_risk_levels:
        raise ValueError(
            f"Invalid risk_level '{data['risk_level']}'. "
            f"Expected one of {valid_risk_levels}."
        )

    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence must be between 0 and 1, got {confidence}."
        )
    data["confidence"] = confidence

    return data
