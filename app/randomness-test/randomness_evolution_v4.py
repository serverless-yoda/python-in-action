import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from collections import Counter

class EntropyAutoML:
    def __init__(self, data_path="./randomness.csv"):
        self.df = pd.read_csv(data_path)
        self.df.columns = ['Draw', 'Date', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'B1', 'B2', 'PV']
        self.data = self.df[['V1','V2','V3','V4','V5','V6']].apply(pd.to_numeric).dropna().values

    def calculate_entropy_pool(self, i):
        """Identifies numbers moving from high chaos to predictable patterns."""
        window = self.data[i-150:i]
        scores = []
        for n in range(1, 51):
            # Calculate spacing between occurrences
            idx = np.where(window == n)[0]
            if len(idx) < 5: 
                scores.append((n, 999))
                continue
            gaps = np.diff(idx)
            # Shannon Entropy of the gaps
            probs = np.histogram(gaps, bins=10, density=True)[0]
            entropy = -np.sum(probs * np.log2(probs + 1e-9))
            scores.append((n, entropy))
        
        # Pick numbers with the LOWEST entropy (most 'organized' patterns)
        scores.sort(key=lambda x: x[1])
        return [x[0] for x in scores[:24]]

    def get_features(self, d):
        d = sorted(d)
        return [np.mean(d), np.std(d), np.median(d), sum(1 for x in d if x % 2 == 0)]

    def run_backtest(self, test_steps=500):
        hits_4 = 0
        model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.03)
        
        for i in range(len(self.data) - test_steps, len(self.data) - 1):
            pool = self.calculate_entropy_pool(i)
            
            # 1. Training with 'Pattern Weighting'
            train_window = self.data[max(0, i-400):i-1]
            X_train = [self.get_features(d) for d in train_window]
            y_train = [len(set(train_window[j]) & set(train_window[j+1])) for j in range(len(train_window)-1)]
            
            # Push the weight to 6th power to ignore everything but 4+ hits
            weights = [y**6 if y >= 3 else 1 for y in y_train]
            model.fit(np.array(X_train[:len(y_train)]), np.array(y_train), sample_weight=weights)
            
            # 2. Generation & Prediction
            candidates = [sorted(np.random.choice(pool, 6, replace=False)) for _ in range(15000)]
            preds = model.predict(np.array([self.get_features(c) for c in candidates]))
            
            top_20 = [candidates[idx] for idx in np.argsort(preds)[-20:]]
            
            if max([len(set(t) & set(self.data[i])) for t in top_20]) >= 4:
                hits_4 += 1
        
        print(f"🌀 ENTROPY v48.0 4+ Rate: {(hits_4 / test_steps) * 100:.1f}%")

EntropyAutoML().run_backtest(500)