#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import time
import json
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

# ============================================================
# 1. CONSTANTS
# ============================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API2 = "https://www.metaculus.com/api2"
HTTP = requests.Session()

# Initialise le stockage des résultats pour qu'ils survivent aux reruns
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

# ============================================================
# 2. PROMPT — METACULUS-STYLE KEY FACTORS, DÉTAILLÉ
# ============================================================

SYSTEM_PROMPT = """
You write short, Metaculus-style "Key factors" for forecasting questions.

You MUST return ONLY a single JSON object:
{
  "question_id": <int>,
  "question_title": "<string>",
  "factors": ["<string>", "..."]
}

No preamble. No markdown. No comments. JSON only.

GOAL
- Help forecasters quickly see the 1–5 most important drivers of the outcome.
- Each factor = one concise sentence, high signal, low fluff.

STYLE
- Natural language, one sentence per factor, <= 180 characters.
- No numbering, quotes, emojis, or bullet markers.
- Read like the Key factors chips under a Metaculus question.
- Example style:
  - "Most US voters still rank inflation, immigration and governance above technology in 'most important problem' polls."
  - "Technology issues remain under 1% of Gallup 'most important problem' responses despite several years of intense AI media coverage."
  - "Large, well-publicised AI incidents (e.g. lethal misuse or major economic shock) could abruptly raise AI salience into mid-single-digit percentages."

SUBSTANCE
- Each factor should describe an important driver, constraint, or baseline that actually moves the forecast.
- Prefer concrete quantities, trends, or comparisons over vague talk.
- You may mention:
  - historical baselines (e.g. past poll shares, past growth rates),
  - relative magnitudes (X >> Y),
  - institutional behaviour (regulators, labs, governments),
  - clear catalysts or blockers (legislation, breakthroughs, crises).
- Avoid generic or tautological statements like:
  - "Geopolitics will matter."
  - "Public opinion is uncertain."
  - "Many things can happen."

EXAMPLE 1 (ILLUSTRATIVE)

Question:
  "What percentage of Americans will consider AI or advancement of computers/technology
   to be the most important problem in January 2028?"

Good factors (illustrative only):
  - "Economy, immigration and government dominate issue salience and leave very little room for technology topics in 'most important problem' polling."
  - "Historically, all technology-related categories sum to well under 1% of responses, even during periods of intense tech news coverage."
  - "Issue salience tends to be sticky; large shifts usually require visible personal hardship (e.g. unemployment, inflation, war) rather than abstract future risks."
  - "A major AI-related disaster or scandal that directly affects many people could temporarily push AI/tech into low-single-digit percentages."

EXAMPLE 2 (ILLUSTRATIVE)

Question:
  "Will Elon Musk be the world's richest person on December 31, 2025?"

Possible factors:
  - "Musk's wealth is dominated by Tesla's market cap; large swings in Tesla stock have outsized impact compared to more diversified billionaires."
  - "Rivals' wealth is spread across multiple assets; catching up would likely require both a Tesla crash and a big rally in competitors' holdings."
  - "Macroeconomic downturns tend to hit high-beta tech stocks harder, increasing downside risk for Musk's net worth leadership."

WHAT TO OUTPUT

- 3 to 6 factors in total.
- Each factor:
  - one sentence,
  - <= 180 characters,
  - concretely informative and obviously relevant to the question.
- Do NOT repeat the question verbatim.
- Do NOT include URLs.
- Return ONLY the JSON object with fields: question_id, question_title, factors.
"""


# ============================================================
# 3. HELPERS
# ============================================================

def get_api_key() -> str:
    key = st.session_state.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    return key


def fetch_question(qid: int) -> Optional[Dict[str, Any]]:
    try:
        r = HTTP.get(f"{API2}/questions/{qid}/", timeout=20)
        r.raise_for_status()
        q = r.json()
        return {
            "id": q["id"],
            "title": q.get("title", f"Question {qid}"),
            "url": q.get("page_url") or f"https://www.metaculus.com/questions/{qid}/",
        }
    except Exception as e:
        st.error(f"Error fetching {qid}: {e!r}")
        return None


def parse_json_strict(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1:
            return json.loads(s[a:b+1])
        raise


def call_llm(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    """Call OpenRouter with required headers to avoid 401."""
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "https://metaculus-factors.local",
        "X-Title": "MetaculusFactors",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.15,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for _ in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(1.2)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return parse_json_strict(content)
        except Exception as e:
            last_err = e
            time.sleep(0.7)

    st.error(f"LLM call failed: {last_err!r}")
    return {"question_id": None, "question_title": "", "factors": []}


def messages_for(qid: int, title: str, url: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"QUESTION_ID: {qid}\n"
                f"TITLE: {title}\n"
                f"URL: {url}\n\n"
                "Write Metaculus-style key factors as specified. JSON only."
            ),
        },
    ]


def rows_from(subject: Dict[str, Any], factors: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for i, f in enumerate(factors, 1):
        rows.append(
            {
                "market_id": subject["id"],
                "title": subject["title"],
                "factor_rank": i,
                "factor": f,
                "url": subject["url"],
            }
        )
    return rows


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ============================================================
# 4. STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Metaculus Key Factors", page_icon="📈", layout="wide")

st.title("📈 Metaculus Question Factors – LLM (IDs only)")
st.caption("Generate Metaculus-style 'Key factors' from question IDs.")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("🔐 OpenRouter API Key")

    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state["OPENROUTER_API_KEY"] = ""

    key_in = st.text_input("Paste API key", type="password")

    colk1, colk2 = st.columns(2)
    with colk1:
        if st.button("Use key"):
            st.session_state["OPENROUTER_API_KEY"] = key_in.strip()
    with colk2:
        if st.button("Clear key"):
            st.session_state["OPENROUTER_API_KEY"] = ""

    if st.session_state.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY"):
        st.success("API key detected.")
    else:
        st.info("No API key yet.")

    st.subheader("🧠 Model (free text)")
    model_choice = st.text_input(
        "OpenRouter model",
        value="openai/gpt-4.1-mini",
        key="model",
        help="Type ANY OpenRouter model id (e.g. openai/gpt-4o, deepseek/deepseek-v3).",
    )

    st.subheader("🔁 Session")
    if st.button("New run / reset"):
        st.session_state["ids_raw"] = ""
        st.session_state["results_df"] = None
        st.experimental_rerun()


# ---------------- Main: IDs input + run button ----------------
ids_raw = st.text_input(
    "Metaculus question IDs (comma-separated):",
    placeholder="38418, 12024, 1068",
    key="ids_raw",
)

go = st.button("▶️ Generate factors", type="primary")

if go:
    # 1) Vérifier la clé
    try:
        _ = get_api_key()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # 2) Parser les IDs (espaces ignorés, ",1068" ok)
    ids: List[int] = []
    for piece in (ids_raw or "").split(","):
        s = piece.strip()
        if not s:
            continue
        if s.isdigit():
            ids.append(int(s))
        else:
            st.warning(f"Ignoring invalid ID: {s!r}")
    if not ids:
        st.error("No valid numeric IDs.")
        st.stop()

    # 3) Récupérer les questions
    subjects: List[Dict[str, Any]] = []
    for qid in ids:
        q = fetch_question(qid)
        if q:
            subjects.append(q)
    if not subjects:
        st.error("Could not fetch any questions.")
        st.stop()

    # 4) Lancer le LLM une fois par question
    all_rows: List[Dict[str, Any]] = []
    prog = st.progress(0.0)

    for i, subj in enumerate(subjects, 1):
        st.info(f"[{i}/{len(subjects)}] {subj['title']}")
        resp = call_llm(messages_for(subj["id"], subj["title"], subj["url"]), model_choice)

        facs_raw = resp.get("factors") or []
        factors: List[str] = []
        if isinstance(facs_raw, list):
            for f in facs_raw:
                if isinstance(f, str):
                    txt = f.strip()
                    if txt:
                        factors.append(txt)

        all_rows.extend(rows_from(subj, factors))
        prog.progress(i / len(subjects))

    if not all_rows:
        st.warning("No factors returned.")
        st.session_state["results_df"] = None
    else:
        df = pd.DataFrame(
            all_rows,
            columns=["market_id", "title", "factor_rank", "factor", "url"],
        )
        st.session_state["results_df"] = df


# ---------------- Persistent results display + CSV ----------------
if st.session_state.get("results_df") is not None:
    df = st.session_state["results_df"]
    st.success(
        f"Generated {len(df)} factors for {df['market_id'].nunique()} question(s) "
        f"using model `{st.session_state.get('model', 'N/A')}`."
    )
    st.dataframe(df, use_container_width=True)

    csv_bytes = df_to_csv_bytes(df)
    # le download ne déclenche qu’un rerun, les données restent dans session_state
    st.download_button(
        "💾 Download CSV",
        data=csv_bytes,
        file_name="metaculus_key_factors.csv",
        mime="text/csv",
        key="download_csv_button",
    )

st.caption("Paste IDs, set your model, click Generate. You can download the CSV multiple times without losing the results.")

