import os
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from xgboost.spark import SparkXGBClassifier

# 1. Initialize Spark with strictly limited memory to prevent Java Heap OOM
spark = SparkSession.builder \
    .appName("Lotto_Master_Integrated_Final") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# --- CORE FUNCTIONS ---

def get_base_data(df):
    """Creates the base targets and machine state without heavy windowing."""
    for n in range(1, 41):
        target = f"n_{n}"
        df = df.withColumn(target, F.when(F.array_contains(F.array("B1","B2","B3","B4","B5","B6"), n), 1).otherwise(0))
    
    df = df.withColumn("row_idx", F.row_number().over(Window.orderBy("Date")))
    df = df.withColumn("evens", sum([(F.col(f"B{i}") % 2 == 0).cast("int") for i in range(1, 4)]))
    return df.na.fill(0)

def train_ball_logic(df, ball_num):
    """Integrated Logic: Momentum + Gap + Freq + Ensemble Models."""
    w_gap = Window.orderBy("Date").rowsBetween(Window.unboundedPreceding, -1)
    w_freq = Window.orderBy("Date").rowsBetween(-20, -1)
    w_mom = Window.orderBy("Date").rowsBetween(-5, -1)
    
    target = f"n_{ball_num}"
    
    # Feature Engineering (On-the-fly to save RAM)
    temp_df = df.withColumn(f"freq_{ball_num}", F.sum(F.col(target)).over(w_freq)) \
                .withColumn(f"mom_{ball_num}", F.sum(F.col(target)).over(w_mom)) \
                .withColumn(f"gap_{ball_num}", F.col("row_idx") - F.last(F.when(F.col(target) == 1, F.col("row_idx")), True).over(w_gap)) \
                .select("Date", target, f"freq_{ball_num}", f"mom_{ball_num}", f"gap_{ball_num}", "evens") \
                .na.fill(0)
    
    features = [f"gap_{ball_num}", f"freq_{ball_num}", f"mom_{ball_num}", "evens"]
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data_vec = assembler.transform(temp_df)
    
    # Ensemble Models (XGBoost + Random Forest)
    xgb = SparkXGBClassifier(n_estimators=30, max_depth=3, label_col=target).fit(data_vec)
    rf = RandomForestClassifier(numTrees=15, maxDepth=4, labelCol=target).fit(data_vec)
    
    # Probability Extraction for the most recent draw
    latest = data_vec.orderBy(F.desc("Date")).limit(1)
    p_xgb = xgb.transform(latest).select("probability").first()[0].toArray()[1]
    p_rf = rf.transform(latest).select("probability").first()[0].toArray()[1]
    
    return (p_xgb * 0.7) + (p_rf * 0.3)

def render_dashboard(scores, stability_val):
    """Displays the AI report and the 7-Line Wheel."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [ball for ball, _ in ranked[:8]]
    
    # Status Logic
    if stability_val >= 0.7: status, icon = "LOCKED", "🔒"
    elif stability_val >= 0.5: status, icon = "SYNCING", "🔄"
    else: status, icon = "CHAOTIC", "⚠️"
    
    report = []
    report.append("╔" + "═"*58 + "╗")
    report.append(f"║ {'AI MOMENTUM DASHBOARD':^56} ║")
    report.append("╠" + "═"*58 + "╣")
    report.append(f"║ STATUS: {status:<15} {icon} | STABILITY: {stability_val*100:>5.1f}%  ║")
    report.append("╟" + "─"*58 + "╢")
    
    for i, (ball, score) in enumerate(ranked[:10], 1):
        bar = "█" * int(score * 100)
        report.append(f"║ {i:>2}. Ball {ball:<2} | {score:.4f} | {bar:<25} ║")

    report.append("╠" + "═"*58 + "╣")
    report.append(f"║ {'7-LINE ABBREVIATED WHEEL (TOP 8)':^56} ║")
    report.append("╠" + "═"*58 + "╣")
    
    patterns = [[0,1,2,3,4,5], [0,1,2,6,7,3], [0,1,4,5,6,7], [2,3,4,5,6,7], [0,2,4,6,1,3], [1,3,5,7,0,2], [0,3,4,7,1,5]]
    for idx, p in enumerate(patterns, 1):
        line = sorted([top_8[i] for i in p])
        report.append(f"║ Line {idx}: {str(line):<48} ║")
    
    report.append("╚" + "═"*58 + "╝")
    
    final_output = "\n".join(report)
    print(final_output)
    
    with open("momentum_report.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

# --- EXECUTION PIPELINE ---

print("📡 Step 1/3: Ingesting Data...")
raw = spark.read.csv("work/randomness_latest.csv", header=True, inferSchema=True)
base_data = get_base_data(raw)

print("📡 Step 2/3: Training Primary Models (Balls 1-40)...")
primary_scores = {}
for b in range(1, 41):
    primary_scores[b] = train_ball_logic(base_data, b)
    if b % 10 == 0: print(f"   [Progress: {b}/40]")

print("📡 Step 3/3: Running Forward Test (Stability Check)...")
# Get data excluding the very last draw
last_draw_date = raw.select(F.max("Date")).collect()[0][0]
n_minus_1_data = get_base_data(raw.filter(F.col("Date") < last_draw_date))

# Re-test top 10 candidates only to save memory
top_10 = sorted(primary_scores, key=primary_scores.get, reverse=True)[:10]
stable_scores = {b: train_ball_logic(n_minus_1_data, b) for b in top_10}

# Calculate Overlap (Stability)
top_8_now = set(sorted(primary_scores, key=primary_scores.get, reverse=True)[:8])
top_8_then = set(sorted(stable_scores, key=stable_scores.get, reverse=True)[:8])
stability = len(top_8_now & top_8_then) / 8

# Output Final Results
render_dashboard(primary_scores, stability)