import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType

# 1. Configurar argumentos de línea de comandos
parser = argparse.ArgumentParser(description="Entrenar SVM con selección LASSO sobre datos curados de MinIO.")
parser.add_argument(
    "--dataset", 
    type=str, 
    default="dropped", 
    choices=["dropped", "imputed"], 
    help="Elige qué dataset limpio usar: 'dropped' o 'imputed'"
)
args = parser.parse_args()

# Seleccionar la ruta en función del argumento
CURATED_PATH = f"s3a://dpl/curated_{args.dataset}"

# 2. Inicializar Spark con soporte para MinIO (S3A)
spark = SparkSession.builder \
    .appName(f"Train SVM - {args.dataset}") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "dpladmin")) \
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "dpladmin123")) \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print(f"--> Cargando datos desde la capa curated: {CURATED_PATH}")
df_spark = spark.read.parquet(CURATED_PATH)
df = df_spark.toPandas()

# Asegurar que 'match' es entero binario
df['match'] = df['match'].astype(int)

# 3. Selección de variables mediante LASSO (L1)
print("Aplicando selección de variables con LASSO...")
X = df.drop(columns=['match'])
X = pd.get_dummies(X, drop_first=True)
y = df['match']

lasso_cv = LogisticRegressionCV(
    cv=10, 
    penalty='l1', 
    solver='liblinear', 
    scoring='roc_auc', 
    random_state=123
)
lasso_cv.fit(X, y)

coefs = pd.Series(lasso_cv.coef_[0], index=X.columns)
selected_vars = coefs[coefs != 0].index.tolist()

print(f"Variables seleccionadas por LASSO ({len(selected_vars)} en total):")

datos2 = df[['match'] + selected_vars].copy()

# 4. División de los datos en Entrenamiento (70%) y Test (30%)
X_subset = datos2.drop(columns=['match'])
y_subset = datos2['match']

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.3, random_state=456, stratify=y_subset
)

# 5. Oversampling para balancear clases
ros = RandomOverSampler(random_state=2600)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# 6. Entrenamiento del modelo SVM con kernel radial (RBF)
print("Entrenando modelo SVM (RBF)...")
modelo_svm = SVC(kernel='rbf', probability=True, random_state=42)
modelo_svm.fit(X_train_bal, y_train_bal)

# 7. Evaluación
y_pred = modelo_svm.predict(X_test)
y_prob = modelo_svm.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
auc_val = roc_auc_score(y_test, y_prob)

print(f"--- Métricas del Modelo SVM ({args.dataset}) ---")
print(f"Accuracy: {accuracy:.4f} | Sensibilidad: {sensitivity:.4f} | Especificidad: {specificity:.4f} | AUC: {auc_val:.4f}")

# 8. Guardar métricas en MinIO con nombre dinámico según el dataset elegido
metrics_data = {
    "model": "SVM",
    "dataset_source": args.dataset,
    "kernel": "rbf",
    "num_features_selected": len(selected_vars),
    "selected_features": selected_vars,
    "metrics": {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc_val),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    }
}

metrics_json_str = json.dumps(metrics_data, indent=4)
METRICS_PATH = f"s3a://dpl/metrics/svm_metrics_{args.dataset}.json"

print(f"Guardando métricas en MinIO: {METRICS_PATH} ...")

sc = spark.sparkContext
rdd = sc.parallelize([metrics_json_str])
df_metrics_out = spark.createDataFrame(rdd, StringType()).toDF("json_data")
df_metrics_out.write.mode("overwrite").text(f"s3a://dpl/metrics/svm_output_temp_{args.dataset}")

print("¡Proceso finalizado y métricas guardadas con éxito en MinIO!")