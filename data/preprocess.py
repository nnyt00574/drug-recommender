import pandas as pd
import numpy as np
from utils.reaction_labels import REACTION_CATEGORIES


def extract_reaction_vector(reactions):
    vec = [0] * len(REACTION_CATEGORIES)

    for r in reactions:
        r = str(r).lower()
        for i, (_, keywords) in enumerate(REACTION_CATEGORIES.items()):
            if any(k in r for k in keywords):
                vec[i] = 1

    return vec


def build_dataset(records):
    data = []

    for r in records:
        try:
            drugs = [
                d.get("medicinalproduct", "").upper()
                for d in r.get("patient", {}).get("drug", [])
                if d.get("medicinalproduct")
            ]

            if len(drugs) < 2:
                continue

            reaction_list = [
                rx.get("reactionmeddrapt", "")
                for rx in r.get("patient", {}).get("reaction", [])
            ]

            reaction_vec = extract_reaction_vector(reaction_list)
            
            
            raw_severity = r.get("serious", 0)
            try:
                severity = float(raw_severity) if raw_severity is not None else 0.0
            except ValueError:
                severity = 0.0

            for i in range(len(drugs)):
                for j in range(i + 1, len(drugs)):
                    data.append({
                        "drug_a": drugs[i],
                        "drug_b": drugs[j],
                        "reaction_vec": reaction_vec,
                        "severity": severity
                    })

        except Exception as e:
            # Catching general exceptions here just in case a record is deeply malformed
            continue

    # 🚨 SAFETY CHECK
    if len(data) == 0:
        raise RuntimeError("No valid drug pairs extracted — check API data")

    df = pd.DataFrame(data)

    # ✅ ensure columns exist
    required_cols = {"drug_a", "drug_b", "reaction_vec", "severity"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"Missing columns: {required_cols - set(df.columns)}")

    # ✅ CREATE PAIR (SAFE)
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["drug_a"], r["drug_b"]])),
        axis=1
    )

    # =============================
    # SAFE GROUPING
    # =============================
    grouped = df.groupby("pair")

    rows = []

    for pair, group in grouped:
        try:
            vecs = np.vstack(group["reaction_vec"].values)
            combined_vec = np.max(vecs, axis=0)

            rows.append({
                "pair": pair,
                "reaction_vec": combined_vec,
                "severity": group["severity"].mean()
            })
        except Exception as e:
            # ✅ FIX: Log the error instead of silently swallowing it
            print(f"⚠️ Error aggregating pair {pair}: {e}")
            continue

    if len(rows) == 0:
        raise RuntimeError("Aggregation failed — no valid grouped rows")

    agg = pd.DataFrame(rows)

    # ✅ SPLIT PAIR
    agg["drug_a"] = agg["pair"].apply(lambda x: x[0])
    agg["drug_b"] = agg["pair"].apply(lambda x: x[1])

    agg = agg.drop(columns=["pair"])

    return agg