from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

# Create Spark session
spark = SparkSession.builder \
    .appName("COVID Classification Analysis") \
    .config("spark.sql.broadcastTimeout", "1200s") \
    .getOrCreate()

# Load dataset
df = spark.read.option("header", True).csv("bioinfo_data/Covid Data.csv")

# Cast all columns to double except the label (assume last column ICU is the label)
columns_to_cast = [c for c in df.columns if c != "ICU"]
for c in columns_to_cast:
    df = df.withColumn(c, col(c).cast("double"))
df = df.withColumn("ICU", col("ICU").cast("int"))

# Drop any rows with nulls
df = df.dropna()

# Define features and label
feature_cols = [c for c in df.columns if c != "ICU"]
label_col = "ICU"

# Build ML pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
selector = ChiSqSelector(featuresCol="features", outputCol="selectedFeatures", labelCol=label_col, numTopFeatures=15)
scaler = StandardScaler(inputCol="selectedFeatures", outputCol="scaledFeatures")
rf = RandomForestClassifier(labelCol=label_col, featuresCol="scaledFeatures", numTrees=20, maxDepth=10)
pipeline = Pipeline(stages=[assembler, selector, scaler, rf])

# Split data
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Train
start_time = time.time()
model = pipeline.fit(train_df)
training_time = time.time() - start_time

# Predict
predictions = model.transform(test_df)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print results
print("\n===== COVID Dataset Spark Results =====")
print(f"Training Time: {training_time:.4f} seconds")
print(f"Test Accuracy: {accuracy:.4f}")
predictions.groupBy("ICU", "prediction").count().show()

# Stop
spark.stop()
