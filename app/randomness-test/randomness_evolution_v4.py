import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from collections import Counter

class PCAAidAutoML:
    def __init__(self, data_path="./randomness.csv"):
        self.df = pd.read_csv(data_path)
        self.df.columns = ['Draw', 'Date', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'B1', 'B2', 'PV']
        self.data = self.df[['V1','V2','V3','V4','V5','V6']].apply(pd.to_numeric).dropna().values

    def get_raw_feats(self, draw):
        d = sorted(draw)
        gaps = np.diff(d)
        return [
            np.mean(d), np.std(d), np.median(d), sum(d),
            sum(1 for x in d if x % 2 == 0), # Parity
            gaps.min(), gaps.max(), np.mean(gaps),
            d[0], d[-1] # High/Low bounds
        ]

    def run_backtest(self, test_steps=500):
        hits_4 = 0
        pca = PCA(n_components=3)
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
        
        for i in range(len(self.data) - test_steps, len(self.data) - 1):
            # 1. Prepare Training Data
            train_window = self.data[i-250:i-1]
            raw_X = [self.get_raw_feats(train_window[j]) for j in range(len(train_window)-1)]
            y_train = [len(set(train_window[j]) & set(train_window[j+1])) for j in range(len(train_window)-1)]
            
            # 2. Apply PCA to find the 'Signal'
            pca.fit(raw_X)
            X_reduced = pca.transform(raw_X)
            model.fit(X_reduced, y_train)
            
            # 3. Predict on 5,000 Candidates from a 26-number Pool
            pool = [int(n) for n, c in Counter(self.data[i-120:i].flatten()).most_common(26)]
            candidates = [sorted(np.random.choice(pool, 6, replace=False)) for _ in range(5000)]
            
            cand_raw = [self.get_raw_feats(c) for c in candidates]
            cand_reduced = pca.transform(cand_raw)
            preds = model.predict(cand_reduced)
            
            top_20 = [candidates[idx] for idx in np.argsort(preds)[-20:]]
            
            if max([len(set(t) & set(self.data[i])) for t in top_20]) >= 4:
                hits_4 += 1
        
        print(f"📉 PCA AutoML v35.0 4+ Rate: {(hits_4 / test_steps) * 100:.1f}%")

PCAAidAutoML().run_backtest(500)