import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class Ensemble:
    def __init__(self):
        self.models = [
            xgb.XGBClassifier(n_estimators=200),
            lgb.LGBMClassifier(),
            RandomForestClassifier()
        ]
        self.meta = LogisticRegression()

    def fit(self,X,y):
        preds=[]
        for m in self.models:
            m.fit(X,y)
            preds.append(m.predict_proba(X)[:,1])
        self.meta.fit(np.column_stack(preds), y)

    def predict(self,X):
        preds=[m.predict_proba(X)[:,1] for m in self.models]
        return self.meta.predict_proba(np.column_stack(preds))[:,1]