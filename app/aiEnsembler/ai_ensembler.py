import os
from multiprocessing.pool import ThreadPool
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from xgboost.spark import SparkXGBClassifier

# 1. Initialize Spark & Suppress Logs
spark = SparkSession.builder.appName("Lotto_NZ_Parallel_Master").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- CORE FUNCTIONS ---

def build_features(df):
    """Optimized Feature Engineering: Gaps, Frequency, and Machine State"""
    w_gap = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, -1)
    w_freq = Window.orderBy("Date").rowsBetween(-20, -1)
    df = df.withColumn("row_idx", F.row_number().over(Window.orderBy("Date")))

    for n in range(1, 41):
        target = f"n_{n}"
        df = df.withColumn(target, F.when(F.array_contains(F.array("B1","B2","B3","B4","B5","B6"), n), 1).otherwise(0))
        df = df.withColumn(f"freq_{n}", F.sum(F.col(target)).over(w_freq))
        last_hit = F.last(F.when(F.col(target) == 1, F.col("row_idx")), True).over(w_gap)
        df = df.withColumn(f"gap_{n}", F.col("row_idx") - last_hit)

    edges = [1, 2, 3, 4, 5, 36, 37, 38, 39, 40]
    edge_hits = sum([F.col(f"B{i}").isin(edges).cast("int") for i in range(1, 4)])
    core_hits = sum([F.col(f"B{i}").between(15, 25).cast("int") for i in range(1, 4)])
    
    df = df.withColumn("cluster_density", F.abs(edge_hits - core_hits))
    df = df.withColumn("evens", sum([(F.col(f"B{i}") % 2 == 0).cast("int") for i in range(1, 4)]))
    df = df.withColumn("inv_val", 3 / (F.col("B1") + F.col("B2") + F.col("B3")))
    df = df.withColumn("moving_harmonic_mean", F.avg("inv_val").over(Window.orderBy("Date").rowsBetween(-10, -1)))

    return df.na.fill(0)

def train_ensemble_pro(df, ball_num):
    """High-Depth Ensemble Model"""
    label = f"n_{ball_num}"
    feature_cols = [f"gap_{ball_num}", f"freq_{ball_num}", "cluster_density", "evens", "moving_harmonic_mean"]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    train_df = assembler.transform(df)
    
    xgb = SparkXGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, label_col=label).fit(train_df)
    rf = RandomForestClassifier(numTrees=100, maxDepth=8, labelCol=label).fit(train_df)
    
    latest = train_df.orderBy(F.desc("Date")).limit(1)
    p_xgb = xgb.transform(latest).select("probability").first()[0].toArray()[1]
    p_rf = rf.transform(latest).select("probability").first()[0].toArray()[1]
    
    return (p_xgb * 0.8) + (p_rf * 0.2)

def get_ball_score(ball_num, data_frame):
    """Worker for Parallel Training"""
    return ball_num, train_ensemble_pro(data_frame, ball_num)

def walk_forward_validation(df, draws=5):
    """Reliability check: Top 10 prediction capture rate"""
    actual_hits = 0
    recent_data = df.orderBy(F.desc("Date")).limit(draws).collect()
    for d in range(draws):
        test_df = df.limit(df.count() - (d + 1))
        ground_truth = [recent_data[d][f"B{i}"] for i in range(1, 7)]
        # Sample for speed
        sample_balls = [10, 20, 30, 40] 
        scores = {b: train_ensemble_pro(test_df, b) for b in sample_balls}
        if any(b in ground_truth for b, s in scores.items() if s > 0.18):
            actual_hits += 1
    return actual_hits / draws

def display_dashboard(val_acc, ball_results, conv_score, recommendation):
    """Final Output Dashboard"""
    ranked = sorted(ball_results.items(), key=lambda x: x[1], reverse=True)
    top_8 = [ball for ball, score in ranked[:8]]
    state = "OVERFIT" if val_acc > 0.85 else "SYNCED" if val_acc >= 0.70 else "CHAOTIC"
    
    print("\n" + "="*60)
    print("             AI LOTTO ENSEMBLE DASHBOARD")
    print("="*60)
    print(f"MODEL RELIABILITY: {val_acc*100:.1f}%")
    print(f"MACHINE STATE:     {state}")
    print(f"CONVICTION SCORE:  {conv_score:.1f}%")
    print(f"RECOMMENDATION:    {recommendation}")
    print("-" * 60)
    
    print(f"{'Rank':<5} | {'Ball':<5} | {'Score':<8} | {'Interpretation'}")
    for i, (ball, score) in enumerate(ranked[:10], 1):
        status = "EXTREME DUE" if score > 0.22 else "HOT SIGNAL" if score > 0.15 else "NEUTRAL"
        print(f"{i:<5} | {ball:<5} | {score:.4f} | {status}")

    print("\n" + "="*60)
    print("           FINAL 7-LINE ABBREVIATED WHEEL")
    print("="*60)
    patterns = [[0,1,2,3,4,5], [0,1,2,6,7,3], [0,1,4,5,6,7], [2,3,4,5,6,7], [0,2,4,6,1,3], [1,3,5,7,0,2], [0,3,4,7,1,5]]
    for i, p in enumerate(patterns, 1):
        line = sorted([top_8[idx] for idx in p])
        print(f"Line {i}: {', '.join(map(str, line))}")
    print("="*60 + "\n")

# --- EXECUTION ---

raw_data = spark.read.csv("work/randomness_latest.csv", header=True, inferSchema=True)
pool = ThreadPool(8) # Parallel processing lanes

# RUN 1: Full Data
print("🚀 Analyzing Machine State (Run 1/2)...")
proc_1 = build_features(raw_data).cache()
res_1 = dict(pool.starmap(get_ball_score, [(i, proc_1) for i in range(1, 41)]))
top_8_r1 = sorted(res_1, key=res_1.get, reverse=True)[:8]

# RUN 2: Remove Last Draw (N-1)
print("🚀 Verifying Stability (Run 2/2)...")
last_date = raw_data.select(F.max("Date")).collect()[0][0]
raw_minus_1 = raw_data.filter(F.col("Date") < last_date)
proc_2 = build_features(raw_minus_1).cache()
res_2 = dict(pool.starmap(get_ball_score, [(i, proc_2) for i in range(1, 41)]))
top_8_r2 = sorted(res_2, key=res_2.get, reverse=True)[:8]

# Reliability & Conviction
print("📊 Calculating Final Metrics...")
val_acc = walk_forward_validation(proc_1, draws=5)
common = set(top_8_r1) & set(top_8_r2)
stability = len(common) / 8
conv_score = ((val_acc + stability) / 2) * 100
rec = "STRONG PLAY" if conv_score > 75 else "CAUTIOUS" if conv_score > 55 else "PASS"

display_dashboard(val_acc, res_1, conv_score, rec)
pool.close()
pool.join()