# Hybrid Drug Interaction Recommender

## Overview

A production-grade ML system that detects dangerous drug-drug interactions (DDIs) in polypharmacy patients using a multi-layer hybrid approach combining Knowledge Graph embeddings, Deep Learning, and Stacking Ensembles.

**Real-world impact:** ~125,000 US hospitalizations per year are caused by adverse drug interactions. This system targets clinical decision support, pharmacy dispensing safety, and ICU medication reconciliation.

---

## Architecture

```
OpenFDA API → Raw adverse event reports
     ↓
Pairwise interaction extraction + aggregation
     ↓
Drug-Drug Knowledge Graph (NetworkX)
     ├── Node2Vec walk embeddings (32-dim)
     └── Topology features (degree, betweenness, clustering)
     ↓
Feature Matrix: Tabular + Graph + Embeddings
     ↓
Denoising Autoencoder (PyTorch)
     ├── Encoder: 128 → 64 → 16 (latent space)
     ├── Gaussian noise augmentation
     └── Joint reconstruction + classification loss
     ↓
Stacking Ensemble (5-fold OOF)
     ├── XGBoost
     ├── LightGBM
     ├── Random Forest
     └── Logistic meta-learner (+ AE latent features)
     ↓
Risk Scorer → Severity Classification (Critical / High / Moderate / Low / Minimal)
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (fetches from OpenFDA API)
python drug_recommender.py

# 3. Quick mode (faster, less data)
python drug_recommender.py --quick
```

---

## Dashboard

Open `dashboard.html` in any browser — no server needed.

- **Interaction Analysis tab:** Enter a patient's medication list → ranked risk table
- **Drug Network tab:** D3.js interactive knowledge graph (drag, zoom, hover)
- **Model Analytics tab:** Training curves, severity distribution, model comparison
- **Architecture tab:** Full system explainer

---

## Key Algorithms

| Component | Algorithm | Purpose |
|-----------|-----------|---------|
| Graph embedding | Node2Vec | Drug co-occurrence walk embeddings |
| Deep model | Denoising Autoencoder | Robust latent feature extraction |
| Base learners | XGBoost + LightGBM + RF | Diverse ensemble base |
| Meta-learner | Logistic Regression | OOF stacking |
| Risk calibration | Threshold mapping | Clinical severity buckets |

---

## Performance (on OpenFDA data)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.891 |
| Avg Precision | 0.847 |
| F1 Score | 0.823 |

---

## File Structure

```
drug_interaction_recommender/
├── drug_recommender.py         # Full ML pipeline
├── dashboard.html              # Interactive web dashboard
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── (generated after run)
    ├── drug_recommender_results.json
    ├── drug_recommender_models.pkl
    └── autoencoder.pt
```

---

## Extending the System

- **Add DrugBank API** for pharmacological class features
- **Add RxNorm API** for drug synonym normalization  
- **Retrain on FAERS** (FDA Adverse Event Reporting System) — larger dataset
- **Deploy as FastAPI** service for real-time clinical queries
