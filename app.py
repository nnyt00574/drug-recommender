import streamlit as st
import json
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Drug AI Dashboard", layout="wide")

# =========================
# STYLE (PRO LOOK)
# =========================
st.markdown("""
<style>
.big-title {
    font-size:28px !important;
    font-weight:600;
}
.card {
    padding:15px;
    border-radius:10px;
    background:#f5f7fa;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">💊 Drug Interaction Intelligence System</p>', unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
DATA_PATH = "outputs/drug_data.json"

if not os.path.exists(DATA_PATH):
    st.error("Run main.py first")
    st.stop()

with open(DATA_PATH) as f:
    data = json.load(f)

df = pd.DataFrame(data["samples"])
symptoms = data["symptoms"]

# reaction labels (edit if needed)
LABELS = ["bleeding", "liver", "cardiac", "allergic", "gastro"]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔍 Search")

mode = st.sidebar.radio("Mode", ["Drug Pair", "Symptom"])

query = st.sidebar.text_input("Enter drug or symptom")

# =========================
# FUNCTION: RISK DISPLAY
# =========================
def show_risk(vec):
    for i, val in enumerate(vec):
        if val == 1:
            st.markdown(f"🔴 **{LABELS[i].upper()} RISK**")
        else:
            st.markdown(f"🟢 {LABELS[i]}")

# =========================
# DRUG SEARCH
# =========================
if mode == "Drug Pair" and query:
    q = query.upper()

    matches = df[
        (df["drug_a"] == q) | (df["drug_b"] == q)
    ]

    if matches.empty:
        st.warning("No interactions found")
    else:
        st.subheader(f"⚠️ Interactions for {q}")

        for _, row in matches.head(10).iterrows():
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.write(f"💊 {row['drug_a']} + {row['drug_b']}")

                # risk vector
                vec = row["reaction_vec"]

                show_risk(vec)

                # bar chart
                chart_df = pd.DataFrame({
                    "risk": LABELS,
                    "value": vec
                })
                st.bar_chart(chart_df.set_index("risk"))

                st.markdown('</div>', unsafe_allow_html=True)

# =========================
# SYMPTOM SEARCH
# =========================
elif mode == "Symptom" and query:
    q = query.lower()

    if q in symptoms:
        st.subheader(f"💊 Drugs linked to '{q}'")

        drugs = symptoms[q]

        col1, col2 = st.columns([2,1])

        col1.write(drugs)

        chart_df = pd.DataFrame({
            "drug": drugs,
            "rank": list(range(len(drugs), 0, -1))
        })

        col2.bar_chart(chart_df.set_index("drug"))

    else:
        st.warning("Symptom not found")

# =========================
# OVERVIEW
# =========================
st.divider()

st.subheader("📊 System Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Interactions", len(df))
col2.metric("Unique Drugs", len(set(df["drug_a"]).union(df["drug_b"])))
col3.metric("Symptoms", len(symptoms))

# =========================
# TOP INTERACTIONS
# =========================
st.subheader("🔥 Top Interactions")

st.dataframe(df.head(20))