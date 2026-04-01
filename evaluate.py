import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

from config import *
from data.fetch import fetch_data
from data.preprocess import build_dataset
from graph.graph_builder import build_graph
from graph.node2vec_embed import compute_embeddings
from graph.gnn import train_gnn
from graph.gae import train_gae
from utils.features import build_features

# 🚨 macOS OpenMP Fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def evaluate_pipeline():
    print("📊 Starting Evaluation Pipeline...")

    # 1. Load and process data (same as main.py)
    records = fetch_data(MAX_RECORDS)
    df = build_dataset(records)
    G = build_graph(df)
    
    emb = compute_embeddings(G, EMB_DIM)
    gnn_emb = train_gnn(G, emb)
    gae_emb = train_gae(G, emb)
    
    X, y = build_features(df, emb, gnn_emb, gae_emb)

    # ✅ 2. Split the data! 80% Training, 20% Testing
    print(f"Splitting data: {len(X)} total samples...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the model ONLY on the training data
    print("Training Random Forest on 80% of data...")
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, n_jobs=1))
    model.fit(X_train, y_train)

    # 4. Predict on the 20% unseen test data
    print("🔍 Generating predictions on 20% test data...")
    y_pred = model.predict(X_test)

    # 5. Print the Evaluation Report
    print("\n" + "="*50)
    print("MODEL EVALUATION REPORT")
    print("="*50)
    
    # Calculate exact subset accuracy (exact matches across all multi-outputs)
    exact_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Exact Accuracy: {exact_accuracy * 100:.2f}%\n")
    
    # Print detailed precision/recall/F1 for each reaction category
    print("Detailed Classification Report (By Reaction Category):")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*50)

if __name__ == "__main__":
    evaluate_pipeline()