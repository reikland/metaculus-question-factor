#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metaculus Question Factors – Streamlit App
-----------------------------------------
• Pulls recent Metaculus questions (or specific IDs)
• Calls OpenRouter LLM to generate 3–5 forecasting factors per question
• Displays a table and lets you download a CSV

SETUP (Streamlit Cloud recommended):
1) Add a secret OPENROUTER_API_KEY in your Streamlit project
2) (Optional) Add OPENROUTER_MODEL in secrets or env
3) Deploy – this app never hard‑codes your key
"""

import os
import io
import time
import json
import csv
import hashlib
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st

# =====================================
# 0) CONFIG & CONSTANTS
# =====================================
PREFERRED_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
]

OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
API2              = "https://www.metaculus.com/api2"
API               = "https://www.metaculus.com/api"
UA                = {"User-Agent": "metaculus-question-factors/1.0 (+python-requests)"}
HTTP              = requests.Session()
TITLE             = "Metaculus Question Factors – Streamlit"

# =====================================
# 1) UTILITIES
# =====================================

def _ascii(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)


def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = HTTP.get(url, params=params or {}, headers=UA, timeout=30)
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "1") or 1)
        time.sleep(min(wait, 10))
        r = HTTP.get(url, params=params or {}, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


# =====================================
# 2) METACULUS FETCHERS
# =====================================

@st.cache_data(show_spinner=False, ttl=300)
def fetch_recent_questions(n_subjects: int = 3, page_limit: int = 80) -> List[Dict[str, Any]]:
    data = _get(f"{API2}/questions/", {"status": "open", "limit": page_limit})
    results = data.get("results") or data.get("data") or []

    def ts(q):
        return q.get("open_time") or q.get("created_at") or q.get("scheduled_close_time") or ""

    results.sort(key=ts, reverse=True)
    out = []
    for q in results[:n_subjects]:
        qid = q.get("id")
        if not qid:
            continue
        out.append(
            {
                "id": qid,
                "title": q.get("title", ""),
                "url": q.get("page_url") or q.get("url") or f"https://www.metaculus.com/questions/{qid}/",
            }
        )
    return out


def fetch_question_by_id(qid: int) -> Optional[Dict[str, Any]]:
    try:
        q = _get(f"{API2}/questions/{qid}/")
        if not q or "id" not in q:
            return None
        return {
            "id": q["id"],
            "title": q.get("title", f"Question {qid}"),
            "url": q.get("page_url") or q.get("url") or f"https://www.metaculus.com/questions/{qid}/",
        }
    except Exception as e:
        st.warning(f"Could not fetch question {qid}: {e!r}")
        return None


# =====================================
# 3) OPENROUTER HELPERS
# =====================================

SYSTEM_PROMPT_FACTORS = (
    "You are an expert forecasting analyst. Given a Metaculus question title and URL, "
    "RETURN ONLY a valid JSON object with keys: question_id (int|null), question_title (string), "
    "factors (array). Return between 3 and 5 factors. Each factor must be an object with keys: "
    "'factor' (short phrase, <=10 words), 'rationale' (<=220 chars), and 'confidence' (float between 0.0 and 1.0). "
    "**Rationale MUST be concrete** and include, in <=220 chars, these three items: "
    "(A) one observable indicator to monitor (e.g. 'monthly FX reserves', 'number of senior resignations'); "
    "(B) a suggested data source (e.g. 'IMF reports', 'official press release', 'Reuters'); "
    "(C) a plausible numeric threshold or pattern that would materially change the outlook (e.g. 'reserves drop >10% in 2 months'). "
    "Do NOT include any explanatory text outside the JSON. Do NOT invent clickable URLs — if you name sources, use generic well-known names only. "
    "Return only the JSON object and nothing else."
)

FEWSHOTS_FACTORS = [
    {
        "role": "user",
        "content": "TITLE: Will the NASA Administrator still be in office on December 31, 2025? URL: https://example/123",
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "question_id": 123,
                "question_title": "Will the NASA Administrator still be in office on December 31, 2025?",
                "factors": [
                    {
                        "factor": "Political support",
                        "rationale": "Indicator: public endorsements & key committee statements; Source: Congressional record/major press; Threshold: 2+ committee chairs publicly call for removal -> sharply lowers chances.",
                        "confidence": 0.65,
                    },
                    {
                        "factor": "Agency turnover",
                        "rationale": "Indicator: number of senior resignations in 3 months; Source: agency press releases / Reuters; Threshold: >=2 senior exec resignations within 90 days signals instability.",
                        "confidence": 0.55,
                    },
                    {
                        "factor": "Legal/investigations",
                        "rationale": "Indicator: formal investigations or DOJ referral; Source: Inspector General/DOJ announcements; Threshold: public referral or indictment -> major negative impact.",
                        "confidence": 0.6,
                    },
                ],
            }
        ),
    },
    {"role": "user", "content": "TITLE: Will Country X default on sovereign debt by 2026-12-31? URL: https://example/456"},
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "question_id": 456,
                "question_title": "Will Country X default on sovereign debt by 2026-12-31?",
                "factors": [
                    {
                        "factor": "FX reserves trend",
                        "rationale": "Indicator: FX reserves monthly change; Source: Central bank / IMF; Threshold: reserves fall >12% over 2 months -> materially increases default risk.",
                        "confidence": 0.72,
                    },
                    {
                        "factor": "Fiscal primary balance",
                        "rationale": "Indicator: quarterly primary deficit as % of GDP; Source: Ministry of Finance / IMF; Threshold: persistent primary deficit >3% GDP for two quarters -> elevated risk.",
                        "confidence": 0.68,
                    },
                    {
                        "factor": "Capital flight / sovereign spreads",
                        "rationale": "Indicator: 10y CDS or sovereign bond spread widening; Source: Bloomberg/Reuters; Threshold: CDS widen >300bps in 30 days -> strong warning sign.",
                        "confidence": 0.7,
                    },
                ],
            }
        ),
    },
]

_cache: Dict[str, Dict[str, Any]] = {}


def or_headers() -> Dict[str, str]:
    """Build headers, preferring session key, then secrets/env. Raises if missing."""
    key = (
        st.session_state.get("OPENROUTER_API_KEY", "").strip()
        or (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else "")
        or os.environ.get("OPENROUTER_API_KEY", "").strip()
    )

    if not key:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY. Paste it in the sidebar or add it in Secrets/env."
        )

    title = (
        st.secrets.get("X_TITLE") if hasattr(st, "secrets") else None
    ) or os.environ.get("X_TITLE", TITLE)

    referer = (
        st.secrets.get("REFERER") if hasattr(st, "secrets") else None
    ) or os.environ.get("REFERER", "https://localhost")

    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": _ascii(referer),
        "X-Title": _ascii(title),
        "User-Agent": _ascii("metaculus-question-factors/1.0"),
    }


@st.cache_data(show_spinner=False, ttl=300)
def list_models_clean() -> List[Dict[str, Any]]:
    try:
        r = requests.get(OPENROUTER_MODELS, headers=or_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        ms = data.get("data") or data.get("models") or []
    except Exception as e:
        st.warning(f"Could not list OpenRouter models: {e!r}")
        return []

    out = []
    for m in ms:
        out.append(
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "context_length": m.get("context_length") or m.get("max_context_length"),
                "pricing": m.get("pricing") or {},
                "tags": m.get("tags") or [],
                "arch": m.get("architecture"),
            }
        )
    return out


def pick_model(user_choice: Optional[str] = None) -> str:
    if user_choice:
        return user_choice

    env_model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if env_model:
        return env_model

    ms = list_models_clean()
    if ms:
        ids = {m.get("id"): m for m in ms if m.get("id")}
        for mid in PREFERRED_MODELS:
            if mid in ids:
                return mid
        # cheapest instruct fallback
        best_id, best_price = None, 1e9
        for m in ms:
            id_ = (m.get("id") or "").lower()
            tags = " ".join((m.get("tags") or [])).lower()
            arch = (m.get("arch") or "").lower()
            if ("instruct" in id_) or ("instruct" in tags) or ("instruct" in arch):
                pr = (m.get("pricing") or {})
                p = pr.get("prompt") or pr.get("input") or 0.0
                try:
                    p = float(p) if p else 0.0
                except:
                    p = 0.0
                if p < best_price:
                    best_price, best_id = p, (m.get("id") or "")
        if best_id:
            return best_id
    return PREFERRED_MODELS[0]


def parse_json_strict(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1:
            return json.loads(s[a : b + 1])
        raise


def call_openrouter(messages: List[Dict[str, str]], model: str, max_tokens: int = 450, retries: int = 3) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last = None
    for k in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers(), json=payload, timeout=60)
            if r.status_code == 404:
                raise RuntimeError("404 No endpoints for model")
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", "2") or 2), 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices in response")
            content = ch[0].get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("Empty content")
            return parse_json_strict(content)
        except Exception as e:
            last = e
            time.sleep(0.6 * (k + 1))
    st.error(f"OpenRouter call failed: {last!r}")
    return {"question_id": None, "question_title": "", "factors": []}


# =====================================
# 4) FACTORS PIPELINE
# =====================================

def build_factors_msgs(qid: int, qtitle: str, qurl: str) -> List[Dict[str, str]]:
    u = (
        f"TITLE: {qtitle}\nURL: {qurl}\nQUESTION_ID: {qid}\n\n"
        "Produce JSON: {\"question_id\": int, \"question_title\": str, \"factors\":[{\"factor\":\"...\",\"rationale\":\"...\",\"confidence\":0.0}, ...]}\n"
        "Return 3-5 factors."
    )
    return ([{"role": "system", "content": SYSTEM_PROMPT_FACTORS}] + FEWSHOTS_FACTORS + [{"role": "user", "content": u}])


def get_question_factors_with_llm(qid: int, qtitle: str, qurl: str, model: str) -> Dict[str, Any]:
    text = f"{qid}|{qtitle}"
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = _cache.get(key)
    if cached:
        return cached

    msgs = build_factors_msgs(qid, qtitle, qurl)
    resp = call_openrouter(msgs, model, max_tokens=450)

    # Normalize
    if not isinstance(resp, dict):
        resp = {"question_id": qid, "question_title": qtitle, "factors": []}
    if "question_id" not in resp:
        resp["question_id"] = qid
    if "question_title" not in resp:
        resp["question_title"] = qtitle

    facs = resp.get("factors") or []
    normalized = []
    for f in facs[:5]:
        if not isinstance(f, dict):
            continue
        factor = (f.get("factor") or "").strip()
        rationale = (f.get("rationale") or "").strip()
        try:
            confidence = float(f.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        normalized.append(
            {
                "factor": factor,
                "rationale": rationale,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    resp["factors"] = normalized
    _cache[key] = resp
    return resp


def rows_from_subject(subject: Dict[str, Any], factors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rank, f in enumerate(factors, 1):
        out.append(
            {
                "market_id": subject["id"],
                "title": subject["title"],
                "factor_rank": rank,
                "factor": f.get("factor"),
                "rationale": f.get("rationale"),
                "confidence": f.get("confidence"),
                "url": subject["url"],
            }
        )
    return out


# =====================================
# 5) STREAMLIT UI
# =====================================

st.set_page_config(page_title="Metaculus Question Factors", page_icon="📈", layout="wide")

st.title("📈 Metaculus Question Factors – LLM")
st.caption("Generate concise, concrete forecasting factors for Metaculus questions.")

with st.sidebar:
    st.subheader("🔐 Credentials")

    # Let the user paste an API key directly (stored only in session state)
    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state.OPENROUTER_API_KEY = ""

    def _apply_key():
        # Clear cached model list whenever the key changes
        list_models_clean.clear()

    key_input = st.text_input(
        "OpenRouter API key",
        value=st.session_state.OPENROUTER_API_KEY,
        type="password",
        help="Stored only for this browser session (in Streamlit session_state).",
    )
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Use this key", on_click=_apply_key):
            st.session_state.OPENROUTER_API_KEY = key_input.strip()
    with col_b:
        if st.button("Clear key"):
            st.session_state.OPENROUTER_API_KEY = ""
            list_models_clean.clear()
            st.rerun()

    # Also allow env/secrets as fallback; show status
    has_key = bool(
        st.session_state.OPENROUTER_API_KEY
        or (hasattr(st, "secrets") and st.secrets.get("OPENROUTER_API_KEY"))
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if has_key:
        st.success("API key set (session/env/secrets).")
    else:
        st.info("Paste your OPENROUTER_API_KEY above or set it via Secrets.")


    st.subheader("🧠 Model")
    models = list_models_clean()
    model_ids = [m.get("id") for m in models if m.get("id")]

    # Order: prefer PREFERRED_MODELS first
    preferred_first = [m for m in PREFERRED_MODELS if m in model_ids]
    others = [m for m in model_ids if m not in preferred_first]
    all_opts = preferred_first + others

    default_model = pick_model(None)
    model_choice = st.selectbox("OpenRouter model", options=all_opts or [default_model], index=(all_opts or [default_model]).index(default_model) if (all_opts or [default_model]) else 0)

    def _refresh_models():
        list_models_clean.clear()
        try:
            st.toast("Model list refreshed")
        except Exception:
            pass
    st.button("🔄 Refresh model list", on_click=_refresh_models)

    st.subheader("⚙️ Run mode")
    mode = st.radio("Choose input mode", ["Recent questions", "Specific IDs"], horizontal=True)

    if mode == "Recent questions":
        n = st.number_input("How many recent questions?", min_value=1, max_value=50, value=3, step=1)
    else:
        qids_text = st.text_input("Comma-separated Metaculus IDs", placeholder="e.g., 14016, 13999, 12024")

    st.divider()
    new_run = st.button("🧹 Start a new run", help="Clears results and resets the interface")
    if new_run:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        list_models_clean.clear()
        st.rerun()

# RUN BUTTON
run_clicked = st.button("▶️ Generate factors", type="primary")

# PLACEHOLDERS
summary_ph = st.empty()
results_ph = st.container()

def _save_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

if run_clicked:
    try:
        used_model = pick_model(model_choice)
        if mode == "Recent questions":
            subjects = fetch_recent_questions(n_subjects=int(n), page_limit=80)
        else:
            ids = [int(x.strip()) for x in (qids_text or "").split(",") if x.strip()]
            subjects = []
            for qid in ids:
                s = fetch_question_by_id(qid)
                if s:
                    subjects.append(s)
        if not subjects:
            st.warning("No subjects retrieved. Check your inputs.")
        else:
            rows: List[Dict[str, Any]] = []
            prog = st.progress(0)
            for i, s in enumerate(subjects, 1):
                summary_ph.info(f"Processing {i}/{len(subjects)}: [{s['id']}] {s['title']}")
                resp = get_question_factors_with_llm(s["id"], s["title"], s["url"], used_model)
                facts = resp.get("factors") or []
                rows.extend(rows_from_subject(s, facts))
                prog.progress(i / len(subjects))

            if rows:
                df = pd.DataFrame(rows, columns=[
                    "market_id", "title", "factor_rank", "factor", "rationale", "confidence", "url"
                ])
                with results_ph:
                    st.success(f"Generated factors for {df['market_id'].nunique()} question(s).")
                    st.dataframe(df, use_container_width=True)

                    csv_bytes = _save_csv_bytes(df)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_bytes,
                        file_name="metaculus_question_factors.csv",
                        mime="text/csv",
                    )

            else:
                st.info("No factors returned.")

    except Exception as e:
        st.error(f"Run failed: {e!r}")

st.caption("Tip: Add OPENROUTER_API_KEY in your app secrets (Settings → Secrets) before running.")

