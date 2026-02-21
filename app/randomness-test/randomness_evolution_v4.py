import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from collections import Counter

class BayesianMetaAI:
    def __init__(self, data_path="./randomness.csv"):
        self.df = pd.read_csv(data_path)
        self.df.columns = ['Draw', 'Date', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'B1', 'B2', 'PV']
        self.data = self.df[['V1','V2','V3','V4','V5','V6']].apply(pd.to_numeric).dropna().values

    def run_backtest(self, test_steps=500):
        hits_4 = 0
        # Strategies: 0=Balanced, 1=Clustered, 2=Spread
        strategy_weights = [1.0, 1.0, 1.0] 
        
        for i in range(len(self.data) - test_steps, len(self.data) - 1):
            # 1. Update Pool (Top 26 + 'Momentum' Top 4)
            pool = [int(n) for n, c in Counter(self.data[i-100:i].flatten()).most_common(26)]
            
            # 2. Generate 3 distinct Archetype Batches
            candidates = [sorted(np.random.choice(pool, 6, replace=False)) for _ in range(15000)]
            
            # Categorize candidates
            batch_0 = [c for c in candidates if 120 <= sum(c) <= 160][:1000] # Balanced
            batch_1 = [c for c in candidates if (c[5]-c[0]) < 30][:1000]     # Clustered
            batch_2 = [c for c in candidates if (c[5]-c[0]) > 40][:1000]     # Spread
            
            # 3. Decision Logic: Allocate 20 tickets based on strategy weights
            total_w = sum(strategy_weights)
            alloc = [int((w/total_w) * 20) for w in strategy_weights]
            while sum(alloc) < 20: alloc[np.argmax(strategy_weights)] += 1
            
            # Select tickets from each batch (Simple top-K for this loop)
            final_selection = batch_0[:alloc[0]] + batch_1[:alloc[1]] + batch_2[:alloc[2]]
            
            # 4. Actual Draw & Bayesian Update
            actual = set(self.data[i])
            results = [len(set(t) & actual) for t in final_selection]
            
            if max(results) >= 4:
                hits_4 += 1
            
            # Reward successful strategies for the next loop
            if len(batch_0) > 0 and max([len(set(t) & actual) for t in batch_0[:20]]) >= 3: strategy_weights[0] += 0.5
            if len(batch_1) > 0 and max([len(set(t) & actual) for t in batch_1[:20]]) >= 3: strategy_weights[1] += 0.5
            if len(batch_2) > 0 and max([len(set(t) & actual) for t in batch_2[:20]]) >= 3: strategy_weights[2] += 0.5

        print(f"📊 BAYESIAN META-LOOP v45.0 4+ Rate: {(hits_4 / test_steps) * 100:.1f}%")
        print(f"⚖️ Final Strategy Weights: {strategy_weights}")

# Execute
BayesianMetaAI().run_backtest(500)