import json
import os
import joblib 

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from config import *
from data.fetch import fetch_data
from data.preprocess import build_dataset
from graph.graph_builder import build_graph
from graph.node2vec_embed import compute_embeddings
from graph.gnn import train_gnn
from graph.gae import train_gae
from utils.features import build_features
from utils.symptoms import build_symptoms

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def run():
    print(" Starting advanced pipeline...")
    records = fetch_data(MAX_RECORDS)
    df = build_dataset(records)
    print("Dataset ready:", len(df))

    G = build_graph(df)
    print("Graph built:", G.number_of_nodes(), "nodes")

    emb = compute_embeddings(G, EMB_DIM)
    gnn_emb = train_gnn(G, emb)
    gae_emb = train_gae(G, emb)

    X, y = build_features(df, emb, gnn_emb, gae_emb)

    print("Features:", X.shape)
    print(" Training model (using RandomForest for macOS stability)...")

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=1))
    model.fit(X, y)

    os.makedirs("outputs", exist_ok=True)

    # ==========================================
    # ✅ NEW: SAVE THE MODEL AND EMBEDDINGS
    # ==========================================
    print("Saving Model and Network Embeddings to disk...")
    joblib.dump(model, "outputs/rf_model.joblib")
    joblib.dump({"emb": emb, "gnn_emb": gnn_emb, "gae_emb": gae_emb}, "outputs/embeddings.joblib")

    samples_df = df.copy()
    samples_df["reaction_vec"] = samples_df["reaction_vec"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else x
    )

    out = {
        "samples": samples_df.to_dict("records"),
        "symptoms": build_symptoms(records),
        "reaction_labels": list(range(y.shape[1]))
    }

    with open("outputs/drug_data.json", "w") as f:
        json.dump(out, f)

    print(" SYSTEM COMPLETE")

if __name__ == "__main__":
    run()