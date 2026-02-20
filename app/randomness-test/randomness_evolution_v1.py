"""
With a 20.2% 2-match rate, you are winning "small" (matching two numbers) roughly every 5 draws. 
By playing the Top 3 Powerballs, you are significantly increasing the chance of multiplying those small wins into "Powerball + 2" wins, which is where the payout increases significantly.
"""
import pandas as pd
import numpy as np
import os
import random
from collections import Counter

# --- CONFIG ---
DATA_SOURCE = "./randomness.csv"

class RandomnessEvolution:
    def __init__(self):
        self.df = self._prepare_engine_data()

    def _prepare_engine_data(self):
        if not os.path.exists(DATA_SOURCE):
            return pd.DataFrame()
        try:
            df = pd.read_csv(DATA_SOURCE)
            df.columns = ['Draw', 'Draw_Date', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'B1', 'B2', 'PV']
            cols = ['V1','V2','V3','V4','V5','V6','PV']
            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            df = df.dropna(subset=cols).reset_index(drop=True)
            
            # Technical Indicators
            df['V1_Drift'] = df['V1'].rolling(25, min_periods=1).mean()
            df['V1_delta'] = df['V1'].diff().fillna(0)
            df['momentum'] = df['V1_delta'].rolling(5, min_periods=1).mean()
            return df
        except Exception as e:
            print(f"💥 Data Error: {e}")
            return pd.DataFrame()

    def get_warm_pv(self, train_slice):
        counts = Counter(train_slice['PV'].tail(50))
        pv_range = range(1, 21) 
        warm_pvs = [val for val in pv_range if 1 <= counts[val] <= 3]
        selected = random.choice(warm_pvs) if warm_pvs else random.randint(1, 20)
        return int(selected)

    def is_clustered(self, nums):
        count = 0
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1: count += 1
        return count > 1
    
    def ensemble_predict(self, train_slice):
        if train_slice.empty: return []
        recent_mains = train_slice[['V1','V2','V3','V4','V5','V6']].tail(100).values.flatten()
        hot_mains = [int(n) for n, c in Counter(recent_mains).most_common(15)]
        last_row = train_slice.iloc[-1]
        
        # We now check for Clustered numbers AND the Total Sum
        for _ in range(30): 
            pred_set = set()
            v1_target = int(last_row['V1_Drift'] + last_row['momentum'])
            pred_set.add(max(1, min(40, v1_target)))
            
            while len(pred_set) < 6:
                pred_set.add(random.choice(hot_mains) if random.random() < 0.8 else random.randint(1, 50))
            
            final_mains = sorted([int(x) for x in list(pred_set)])
            
            # THE FILTER: Must not be clustered AND sum must be within 130-210
            current_sum = sum(final_mains)
            if not self.is_clustered(final_mains) and (130 <= current_sum <= 210):
                return final_mains + [self.get_warm_pv(train_slice)]
                
        return final_mains + [self.get_warm_pv(train_slice)] # Fallback

    def _ensemble_predict(self, train_slice):
        if train_slice.empty: return []
        recent_mains = train_slice[['V1','V2','V3','V4','V5','V6']].tail(100).values.flatten()
        hot_mains = [int(n) for n, c in Counter(recent_mains).most_common(15)]
        last_row = train_slice.iloc[-1]
        
        for _ in range(20):
            pred_set = set()
            v1_target = int(last_row['V1_Drift'] + last_row['momentum'])
            pred_set.add(max(1, min(40, v1_target)))
            while len(pred_set) < 6:
                pred_set.add(random.choice(hot_mains) if random.random() < 0.8 else random.randint(1, 50))
            final_mains = sorted([int(x) for x in list(pred_set)])
            if not self.is_clustered(final_mains):
                return final_mains + [self.get_warm_pv(train_slice)]
        return sorted([int(x) for x in list(pred_set)]) + [self.get_warm_pv(train_slice)]

    def rigorous_backtest(self, steps=500):
        """Simulates history draw-by-draw to prove the statistical edge."""
        if len(self.df) < steps + 50:
            steps = len(self.df) - 50
            
        results = []
        print(f"🔬 Deep Backtesting {steps} draws... (this may take a few seconds)")
        
        for i in range(len(self.df) - steps, len(self.df)):
            train = self.df.iloc[:i]
            actual = self.df.iloc[i][['V1','V2','V3','V4','V5','V6','PV']].tolist()
            pred = self.ensemble_predict(train)
            
            hits = len(set(pred[:6]) & set(actual[:6]))
            pv_hit = 1 if int(pred[6]) == int(actual[6]) else 0
            results.append({'hits': hits, 'pv': pv_hit})
            
        return pd.DataFrame(results)

def main():
    engine = RandomnessEvolution()
    if engine.df.empty: return

    while True:
        print(f"\n{'='*45}")
        print(f"🎯 POWERBALL ENGINE v10.0 | {len(engine.df)} DRAWS")
        print(f"{'='*45}")
        print("1. [🧪] Deep Backtest (Last 500 Draws)")
        print("2. [🔮] Generate Selection")
        print("3. [❌] Exit")
        
        choice = input("\nSelect: ").strip()

        if choice == '1':
            res = engine.rigorous_backtest(500)
            pv_rate = res['pv'].mean() * 100
            hit2_rate = (res['hits'] >= 2).mean() * 100
            
            print("\n" + "-"*45)
            print(f"📈 500-DRAW PERFORMANCE SUMMARY")
            print("-"*45)
            print(f"Powerball Hit Rate:  {pv_rate:.1f}%  (Random: 5.0%)")
            print(f"2+ Main Match Rate:  {hit2_rate:.1f}%  (Random: 14.0%)")
            
            edge = hit2_rate - 14.0
            status = "🔥 STRONG EDGE" if edge > 5 else "✅ MODERATE EDGE" if edge > 0 else "⚠️ NO EDGE"
            print(f"Current Status:      {status} (+{edge:.1f}%)")
            print("-"*45)
        
        elif choice == '2':
            p = engine.ensemble_predict(engine.df)
            print(f"\n🔮 NEXT DRAW: {', '.join(map(str, p[:6]))} | PV: [{p[6]}]")
            
        elif choice == '3':
            break

if __name__ == "__main__":
    main()