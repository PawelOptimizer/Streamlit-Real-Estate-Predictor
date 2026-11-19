"""LLM-based explanation of a single price prediction (XAI part of the workshop)."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_LLM_MODEL = "gpt-6-luna"

SYSTEM_PROMPT = (
    "Jesteś analitykiem rynku nieruchomości. Wyjaśniasz użytkownikowi wynik modelu ML, "
    "który przewiduje cenę za m² mieszkania. Opieraj się wyłącznie na przekazanych danych "
    "(parametry mieszkania, predykcja, analiza what-if). Nie wymyślaj nowych liczb."
)

USER_PROMPT_TEMPLATE = """Dane modelu:
- dane treningowe: ogłoszenia z jednego miasta (Poznań), okres {date_min} - {date_max}
- błąd modelu na zbiorze testowym (MAPE): {mape}%

Parametry mieszkania (puste pola pominięto):
{row}

Predykcja ceny za m²: {prediction}

Analiza what-if (zmieniamy jeden parametr, pozostałe bez zmian; delta = zmiana predykcji w PLN/m²):
{effects}

Napisz krótkie wyjaśnienie po polsku (max ok. 150 słów), w punktach:
1. Co miało największy wpływ na cenę.
2. Jakie czynniki podwyższają, a jakie obniżają cenę.
3. Jedno zdanie o ograniczeniach (np. przedział niepewności, wiek danych).
"""


def resolve_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        return ""


def _compact_row(row: dict[str, Any], max_text_len: int = 300) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if v is None or (isinstance(v, float) and v != v) or str(v).strip() == "":
            continue
        s = str(v)
        out[k] = s[:max_text_len] + "..." if len(s) > max_text_len else v
    return out


def explain_prediction(
    *,
    api_key: str,
    model: str,
    row: dict[str, Any],
    prediction: dict[str, Any],
    effects: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    test_metrics = meta.get(f"metrics_{meta.get('model_kind')}") or {}
    prompt = USER_PROMPT_TEMPLATE.format(
        date_min=str(meta.get("date_min", "?"))[:10],
        date_max=str(meta.get("date_max", "?"))[:10],
        mape=round(test_metrics.get("mape_pct", float("nan")), 1),
        row=json.dumps(_compact_row(row), ensure_ascii=False, default=str, indent=1),
        prediction=json.dumps(prediction, ensure_ascii=False),
        effects=json.dumps(effects, ensure_ascii=False, indent=1),
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # Reasoning models count hidden reasoning tokens toward this limit, so keep it generous
        max_completion_tokens=2000,
    )
    return response.choices[0].message.content or ""
