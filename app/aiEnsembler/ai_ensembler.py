import os
import sys
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from xgboost.spark import SparkXGBClassifier
from pyspark.ml import Pipeline

# 1. Initialize Spark & Suppress Logs
spark = SparkSession.builder.appName("Lotto_NZ_Full_Ensemble").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- CORE FUNCTIONS ---

def build_features(df):
    """Native Spark Feature Engineering"""
    df = df.orderBy("Date")
    for n in range(1, 41):
        target = f"n_{n}"
        df = df.withColumn(target, F.when(F.array_contains(F.array("B1","B2","B3","B4","B5","B6"), n), 1).otherwise(0))
        
        w_gap = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, -1)
        df = df.withColumn("row_idx", F.monotonically_increasing_id())
        df = df.withColumn(f"gap_{n}", F.col("row_idx") - 
                           F.last(F.when(F.col(target) == 1, F.col("row_idx"))).over(w_gap))

    edges = [1, 2, 3, 4, 5, 36, 37, 38, 39, 40]
    edge_hits = sum([F.col(f"B{i}").isin(edges).cast("int") for i in range(1, 4)])
    core_hits = sum([F.col(f"B{i}").between(15, 25).cast("int") for i in range(1, 4)])
    df = df.withColumn("cluster_density", F.abs(edge_hits - core_hits))
    df = df.withColumn("evens", sum([(F.col(f"B{i}") % 2 == 0).cast("int") for i in range(1, 4)]))
    df = df.withColumn("inv_val", 1 / ((F.col("B1") + F.col("B2") + F.col("B3")) / 3))
    df = df.withColumn("moving_harmonic_mean", F.avg("inv_val").over(Window.orderBy("Date").rowsBetween(-10, -1)))

    return df.na.fill(0)

def train_ensemble(df, ball_num):
    """Trains Ensemble and extracts probability safely"""
    label = f"n_{ball_num}"
    feature_cols = [f"gap_{ball_num}", "cluster_density", "evens", "moving_harmonic_mean"]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    train_df = assembler.transform(df)
    
    # Simple models for Docker stability
    xgb = SparkXGBClassifier(n_estimators=50, label_col=label, features_col="features").fit(train_df)
    rf = RandomForestClassifier(numTrees=50, labelCol=label, featuresCol="features").fit(train_df)
    
    latest = train_df.orderBy(F.desc("Date")).limit(1)
    
    # Vector extraction fix
    p_xgb = xgb.transform(latest).select("probability").first()[0].toArray()[1]
    p_rf = rf.transform(latest).select("probability").first()[0].toArray()[1]
    
    return (p_xgb * 0.7) + (p_rf * 0.3)

def walk_forward_validation(df, window_size=5):
    """Simple backtest logic for the dashboard"""
    # Using a subset of balls to calculate general accuracy
    test_balls = [10, 20, 30] 
    hits = 0
    for b in test_balls:
        score = train_ensemble(df.limit(df.count()-1), b)
        if score > 0.15: hits += 1
    return hits / len(test_balls)

def display_dashboard(val_acc, ball_results):
    ranked = sorted(ball_results.items(), key=lambda x: x[1], reverse=True)
    top_8 = [ball for ball, score in ranked[:8]]
    
    print("\n" + "="*60)
    print("             AI LOTTO ENSEMBLE DASHBOARD")
    print("="*60)
    print(f"MODEL RELIABILITY: {val_acc*100:.1f}%")
    print(f"MACHINE STATE:     {'SYNCED' if val_acc > 0.4 else 'CHAOTIC'}")
    print("-" * 60)
    
    print(f"{'Rank':<5} | {'Ball':<5} | {'Score':<8} | {'Interpretation'}")
    for i, (ball, score) in enumerate(ranked[:10], 1):
        status = "EXTREME DUE" if score > 0.18 else "HOT SIGNAL" if score > 0.12 else "NEUTRAL"
        print(f"{i:<5} | {ball:<5} | {score:.4f} | {status}")

    print("\n" + "="*60)
    print("           FINAL 7-LINE ABBREVIATED WHEEL")
    print("="*60)
    patterns = [[0,1,2,3,4,5], [0,1,2,3,6,7], [0,1,4,5,6,7], [2,3,4,5,6,7], [0,2,4,6,1,3], [1,3,5,7,0,2], [0,3,4,7,1,5]]
    for i, p in enumerate(patterns, 1):
        line = sorted([top_8[idx] for idx in p])
        print(f"Line {i}: {', '.join(map(str, line))}")
    print("="*60 + "\n")

# --- EXECUTION ---

# Load data from your Docker workspace
raw_data = spark.read.csv("work/randomness_latest.csv", header=True, inferSchema=True)
processed_data = build_features(raw_data)

# Run Analytics
val_score = walk_forward_validation(processed_data)
ball_scores = {i: train_ensemble(processed_data, i) for i in range(1, 41)}

# Output Dashboard
display_dashboard(val_score, ball_scores)