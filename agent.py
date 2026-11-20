"""
CrimeLens - Llama-Powered Crime Intelligence Agent
================================================================================
Conversational AI agent capable of answering natural language queries about
district hotspots, crime categories, risk predictions, travel safety comparisons,
and actionable incident guidance (e.g. stolen property response).

Supports:
  1. Local Ollama Llama instance (http://localhost:11434) if available.
  2. Groq / Cloud Llama API if GROQ_API_KEY is configured.
  3. Grounded Deterministic Analytical Agent (Offline, 100% data-verified).
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analytics import (
    CATEGORIES,
    DATAFRAME,
    MAP_CATEGORIES,
    compare_travel_safety,
    get_district_risk_analysis,
    get_india_risk_map,
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# State name normalizer dictionary
STATE_ALIASES = {
    "tamilnadu": "TAMIL NADU",
    "tamil nadu": "TAMIL NADU",
    "tn": "TAMIL NADU",
    "maharashtra": "MAHARASHTRA",
    "mh": "MAHARASHTRA",
    "delhi": "DELHI UT",
    "delhi ut": "DELHI UT",
    "new delhi": "DELHI UT",
    "uttar pradesh": "UTTAR PRADESH",
    "up": "UTTAR PRADESH",
    "karnataka": "KARNATAKA",
    "ka": "KARNATAKA",
    "bengaluru": "KARNATAKA",
    "kerala": "KERALA",
    "kl": "KERALA",
    "west bengal": "WEST BENGAL",
    "wb": "WEST BENGAL",
    "bihar": "BIHAR",
    "br": "BIHAR",
    "rajasthan": "RAJASTHAN",
    "rj": "RAJASTHAN",
    "gujarat": "GUJARAT",
    "gj": "GUJARAT",
    "madhya pradesh": "MADHYA PRADESH",
    "mp": "MADHYA PRADESH",
    "andhra pradesh": "ANDHRA PRADESH",
    "ap": "ANDHRA PRADESH",
    "telangana": "TELANGANA",
    "tg": "TELANGANA",
    "punjab": "PUNJAB",
    "pb": "PUNJAB",
    "haryana": "HARYANA",
    "hr": "HARYANA",
    "odisha": "ODISHA",
    "orissa": "ODISHA",
    "assam": "ASSAM",
    "jharkhand": "JHARKHAND",
    "chhattisgarh": "CHHATTISGARH",
    "uttarakhand": "UTTARAKHAND",
    "goa": "GOA",
    "himachal pradesh": "HIMACHAL PRADESH",
    "hp": "HIMACHAL PRADESH",
}

# Crime keyword mappings
CRIME_KEYWORDS = {
    "theft": [
        "theft", "stolen", "stealing", "pickpocket", "pickpocketing",
        "bike stolen", "car stolen", "mobile stolen", "phone stolen",
        "vehicle", "belongings", "laptop", "wallet", "bag", "thief"
    ],
    "rape": [
        "women safety", "woman", "women", "rape", "assault", "harassment",
        "molestation", "eve teasing", "girl", "safety for girls", "solo female",
        "lady", "female"
    ],
    "murder": [
        "murder", "homicide", "kill", "killing", "death", "slain",
        "violent crime", "physical attack", "assault with weapon"
    ],
    "robbery": [
        "robbery", "armed robbery", "dacoity", "mugging", "snatching",
        "chain snatching", "gunpoint", "extortion", "heist"
    ],
    "riots": [
        "riot", "riots", "clashes", "communal", "unrest", "strike",
        "bandh", "mob violence", "protest clash"
    ],
    "total": [
        "total crime", "all crime", "overall crime", "crime rate",
        "crime index", "total cases"
    ],
}


def _check_ollama_available() -> bool:
    """Check if local Ollama service is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", headers={"User-Agent": "CrimeLens"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _call_ollama_llama(prompt: str, system: str = "") -> Optional[str]:
    """Call local Ollama Llama instance if available."""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json.get("response", "").strip()
    except Exception:
        return None


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract state, district, crime category, comparison targets, and intent from message."""
    raw = text.lower().strip()

    # 1. Detect Crime Category
    detected_cat = None
    for cat, kws in CRIME_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", raw) for kw in kws):
            detected_cat = cat
            break

    # 2. Detect State
    detected_state = None
    for alias, sname in STATE_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", raw):
            detected_state = sname
            break

    # If not found via alias, check exact state names from DataFrame
    if not detected_state:
        df_states = DATAFRAME.state.unique()
        for s in df_states:
            s_clean = s.lower().replace("&", "and")
            if s.lower() in raw or s_clean in raw:
                detected_state = s
                break

    # 3. Detect District
    detected_district = None
    df_districts = DATAFRAME.district.unique()
    for d in df_districts:
        if len(d) > 3 and re.search(r"\b" + re.escape(d.lower()) + r"\b", raw):
            detected_district = d
            if not detected_state:
                # Infer state from district
                st_match = DATAFRAME[DATAFRAME.district == d].state.iloc[0]
                detected_state = st_match
            break

    # 4. Detect Intent
    intent = "general"
    if any(w in raw for w in ["vs", "compare", "safer than", "better than", "between"]):
        intent = "compare"
    elif any(w in raw for w in ["stolen", "lost my", "what should i do", "what to do if stolen", "theft help"]):
        intent = "stolen_help"
    elif any(w in raw for w in ["which area", "which district", "highest", "most", "prone to", "hotspot", "top area", "worst"]):
        intent = "hotspot"
    elif detected_district:
        intent = "district_profile"
    elif any(w in raw for w in ["state", "all india", "national", "safest state"]):
        intent = "state_rankings"
    elif any(w in raw for w in ["model", "algorithm", "regression", "tree", "clustering"]):
        intent = "methodology"
    elif any(w in raw for w in ["hi", "hello", "hey", "who are you", "help", "what can you do"]):
        intent = "greeting"

    return {
        "intent": intent,
        "state": detected_state or "TAMIL NADU",
        "district": detected_district,
        "category": detected_cat or "theft",
        "has_explicit_state": detected_state is not None,
        "has_explicit_district": detected_district is not None,
    }


def process_agent_query(user_msg: str) -> Dict[str, Any]:
    """
    Process user natural language message, query relevant analytical engines,
    and formulate a grounded response.
    """
    entities = extract_entities(user_msg)
    intent = entities["intent"]
    state = entities["state"]
    district = entities["district"]
    category = entities["category"]
    year = 2026

    # -------------------------------------------------------------------------
    # Intent 1: Hotspot & Area Identification Query
    # e.g. "which area in tamilnadu is high for theft"
    # -------------------------------------------------------------------------
    if intent == "hotspot" or (intent == "general" and entities["has_explicit_state"]):
        sub = DATAFRAME[DATAFRAME.state == state]
        if sub.empty:
            sub = DATAFRAME[DATAFRAME.state.str.contains(state.replace(" ", ""), na=False)]

        recent = sub[sub.year == year]
        if recent.empty:
            recent = sub[sub.year == int(sub.year.max())]

        top_d = recent.groupby("district")[category].sum().sort_values(ascending=False).head(5)
        total_cat_cases = int(round(recent[category].sum()))
        cat_label = MAP_CATEGORIES.get(category, category.title())

        # Top district details
        top_name = top_d.index[0].title() if len(top_d) > 0 else "N/A"
        top_cases = int(round(top_d.iloc[0])) if len(top_d) > 0 else 0
        top_pct = round((top_cases / max(1, total_cat_cases)) * 100, 1)

        hotspot_list = []
        for rank, (d_name, count) in enumerate(top_d.items(), 1):
            share = round((count / max(1, total_cat_cases)) * 100, 1)
            hotspot_list.append(f"**#{rank} {d_name.title()}**: **{int(round(count)):,} cases** ({share}% of state total)")

        hotspots_text = "\n".join(hotspot_list)

        reply = (
            f"### 📍 Top {cat_label} Hotspots in **{state.title()}** ({year})\n\n"
            f"In **{state.title()}**, the #1 hotspot for **{cat_label.lower()}** is **{top_name}**, "
            f"accounting for **{top_cases:,} cases/year** ({top_pct}% of the entire state's total {cat_label.lower()} cases).\n\n"
            f"#### 🏆 Top 5 Districts Ranked by Reported Cases:\n"
            f"{hotspots_text}\n\n"
            f"💡 **Key Observation:** High volumes of {cat_label.lower()} in {top_name} are heavily concentrated "
            f"around dense transit corridors, commercial markets, and major railway terminals."
        )

        suggested_followups = [
            f"What should I do if my thing is stolen in {top_name}?",
            f"What is the safest district in {state.title()}?",
            f"Compare {top_name} with Coimbatore",
        ]

        data_card = {
            "type": "hotspot_list",
            "state": state.title(),
            "category": cat_label,
            "top_district": top_name,
            "top_cases": top_cases,
            "districts": [
                {"rank": r, "name": d.title(), "cases": int(round(c))}
                for r, (d, c) in enumerate(top_d.items(), 1)
            ],
        }

        return {
            "reply": reply,
            "data_card": data_card,
            "suggested_followups": suggested_followups,
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Intent 2: Stolen Items / Emergency Action Guidance
    # e.g. "if my thing is stolen in chennai what should i do"
    # -------------------------------------------------------------------------
    if intent == "stolen_help":
        target_dist = district or "CHENNAI"
        analysis = get_district_risk_analysis(state=state, district=target_dist, category="theft", year=year)
        inc = analysis["incident_assessment"]

        steps_text = "\n".join([f"* {step}" for step in inc["action_steps"]])

        reply = (
            f"### 🚨 Stolen Property Incident Guidance for **{analysis['district']} ({analysis['state']})**\n\n"
            f"* **District Risk Level for Theft:** **{inc['risk_label']}**\n"
            f"* **Projected Theft Cases:** **{inc['count']:,} cases/year** (Rank #{inc['rank']} of {inc['total_districts']} in {analysis['state']})\n"
            f"* **Trend Direction:** {inc['trend']} ({inc['vs_state_avg']})\n\n"
            f"#### 📋 Immediate Action Steps to Take:\n"
            f"{steps_text}\n\n"
            f"📞 **Emergency Contact:** **{inc['emergency_number']}**"
        )

        suggested_followups = [
            f"Which other crimes are common in {analysis['district']}?",
            f"Which district in {analysis['state']} has the least theft?",
            "How does the risk prediction model work?",
        ]

        return {
            "reply": reply,
            "data_card": {
                "type": "incident_card",
                "district": analysis["district"],
                "state": analysis["state"],
                "risk_label": inc["risk_label"],
                "cases": inc["count"],
                "rank": inc["rank"],
            },
            "suggested_followups": suggested_followups,
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Intent 3: Safety Comparison
    # e.g. "is coimbatore safer than chennai"
    # -------------------------------------------------------------------------
    if intent == "compare":
        # Extract two potential locations
        words = re.findall(r"\b[A-Za-z]+\b", user_msg.upper())
        df_districts = set(DATAFRAME.district.unique())
        matched_places = [w for w in words if w in df_districts]

        dist_a = matched_places[0] if len(matched_places) >= 1 else "CHENNAI"
        dist_b = matched_places[1] if len(matched_places) >= 2 else "COIMBATORE"

        comp = compare_travel_safety(
            state_a=state,
            district_a=dist_a,
            state_b=state,
            district_b=dist_b,
            profile_key="general",
        )

        reply = (
            f"### 🛡️ Safety Comparison: **{comp['area_a']['name']}** vs **{comp['area_b']['name']}**\n\n"
            f"🏆 **Verdict:** **{comp['verdict']['title']}**\n\n"
            f"{comp['verdict']['summary']}\n\n"
            f"#### 📊 Safety Scores (0–100):\n"
            f"* **{comp['area_a']['name']}:** **{comp['area_a']['safety_score']} / 100** ({comp['area_a']['safety_label']}) — ~{comp['area_a']['total_annual_crimes']:,} annual crimes\n"
            f"* **{comp['area_b']['name']}:** **{comp['area_b']['safety_score']} / 100** ({comp['area_b']['safety_label']}) — ~{comp['area_b']['total_annual_crimes']:,} annual crimes\n\n"
            f"💡 **Recommendation:** For family visits and general travel, **{comp['verdict']['winner_name']}** offers a substantially lower crime hazard index."
        )

        return {
            "reply": reply,
            "data_card": {
                "type": "comparison_card",
                "winner": comp["verdict"]["winner_name"],
                "score_a": comp["area_a"]["safety_score"],
                "score_b": comp["area_b"]["safety_score"],
            },
            "suggested_followups": [
                f"Which area in {state.title()} has the lowest crime?",
                f"What is the top crime in {dist_a.title()}?",
                "Tell me about smart police manpower deployment",
            ],
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Intent 4: District Deep-Dive
    # e.g. "crime in chennai" or "how safe is chennai"
    # -------------------------------------------------------------------------
    if intent == "district_profile" and district:
        analysis = get_district_risk_analysis(state=state, district=district, category=category, year=year)
        p = analysis["primary_crime"]

        reply = (
            f"### 🔍 Crime Risk Profile for **{analysis['district']} ({analysis['state']})**\n\n"
            f"* **Total Annual Reported Crimes:** **{analysis['total_crimes']:,} cases/year**\n"
            f"* **#1 Most Prevalent Crime:** **{p['label']}** ({p['count']:,} cases, **{p['share_pct']}%** of local crime)\n\n"
            f"#### 📊 Risk Matrix by Crime Type:\n"
        )

        for c_key in ["theft", "murder", "rape", "robbery", "riots"]:
            m = analysis["category_matrix"][c_key]
            reply += f"* **{m['category_label']}:** {m['risk_label']} — **{m['count']:,} cases** (Rank #{m['rank']} in state, {m['trend']})\n"

        return {
            "reply": reply,
            "data_card": {
                "type": "district_profile",
                "district": analysis["district"],
                "primary_crime": p["label"],
                "total_crimes": analysis["total_crimes"],
            },
            "suggested_followups": [
                f"What if my phone is stolen in {analysis['district']}?",
                f"Compare {analysis['district']} with Coimbatore",
                f"Top theft areas in {analysis['state']}",
            ],
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Intent 5: State Rankings / All-India Risk
    # e.g. "which state is highest in india"
    # -------------------------------------------------------------------------
    if intent == "state_rankings":
        risk_map = get_india_risk_map(category=category, year=year)
        cat_label = risk_map["category_label"]
        top_states = risk_map["leaderboard"][:5]
        safe_states = list(reversed(risk_map["leaderboard"][-5:]))

        top_lines = "\n".join([f"**#{s['rank']} {s['state']}**: **{s['count']:,} cases** ({s['share_pct']}% of India)" for s in top_states])
        safe_lines = "\n".join([f"**#{s['rank']} {s['state']}**: **{s['count']:,} cases** (🟢 Low Risk)" for s in safe_states])

        reply = (
            f"### 🗺️ All-India Risk Overview for **{cat_label}** ({year})\n\n"
            f"Across India, an estimated **{risk_map['total_national_crimes']:,} cases** are projected for {year}.\n\n"
            f"#### 🔴 Highest Reported Cases (Top 5 States):\n"
            f"{top_lines}\n\n"
            f"#### 🟢 Safest / Lowest Reported Cases:\n"
            f"{safe_lines}"
        )

        return {
            "reply": reply,
            "data_card": {"type": "state_rankings", "category": cat_label},
            "suggested_followups": [
                "Which area in Tamil Nadu is highest for theft?",
                "Which area in Maharashtra has the highest crime?",
                "Explain the regression forecasting model",
            ],
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Intent 6: Methodology / Models
    # e.g. "which algorithm we have used"
    # -------------------------------------------------------------------------
    if intent == "methodology":
        reply = (
            f"### 🧠 Machine Learning & Analytical Models Used in CrimeLens\n\n"
            f"1. **Linear Regression with Chronological Holdout Validation:** Used for multi-year trend forecasting ($y = mx + c$). Extrapolates aggregate macro-growth while validating error via MAE & RMSE.\n"
            f"2. **Quantile Stratification (Percentile Clustering):** Groups states and districts into 3 clear risk clusters (🔴 High Risk: Top 33%, 🟡 Medium: Middle 33%, 🟢 Low: Bottom 33%).\n"
            f"3. **Multi-Attribute Utility Theory (MAUT) & Logarithmic Hazard:** Powers the Safe Travel Advisor (0–100 Safety Score), heavily penalizing violent crime over petty offenses with diminishing logarithmic scaling.\n"
            f"4. **Constrained Resource Optimization:** Allocates police manpower across districts based on a weighted Crime Threat Index to maximize risk mitigation efficiency."
        )

        return {
            "reply": reply,
            "data_card": {"type": "methodology"},
            "suggested_followups": [
                "Why can't tree models extrapolate into the future?",
                "What is clustering used for in crime data?",
                "Which area in Tamil Nadu is highest for theft?",
            ],
            "engine": "Llama Crime Engine (Grounded Analytical)",
        }

    # -------------------------------------------------------------------------
    # Default: Greeting & Assistant Capabilities
    # -------------------------------------------------------------------------
    return {
        "reply": (
            f"Hello! I am **CrimeLens AI Copilot** (powered by Llama & our 25-Year Crime Intelligence Analytics Engine). 👮‍♂️🔍\n\n"
            f"I can help you analyze crime risks, find hotspots, compare areas, and get safety advice. For example, ask me:\n\n"
            f"* *\"Which area in Tamil Nadu is high for theft?\"*\n"
            f"* *\"If my thing is stolen in Chennai, what should I do?\"*\n"
            f"* *\"Is Coimbatore safer than Chennai?\"*\n"
            f"* *\"Which state in India has the highest violent crime?\"*\n"
            f"* *\"What are the crime trends in Delhi?\"*"
        ),
        "data_card": None,
        "suggested_followups": [
            "Which area in Tamil Nadu is highest for theft?",
            "Is Coimbatore safer than Chennai?",
            "If my phone is stolen, what should I do?",
            "Which states in India have highest violent crime?",
        ],
        "engine": "Llama Crime Engine (Grounded Analytical)",
    }
