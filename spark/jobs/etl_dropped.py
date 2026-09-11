from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

# 1. Inicializar Spark con 4GB de RAM
spark = SparkSession.builder \
    .appName("ETL Speed Dating - Dropped") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
    .config("spark.hadoop.fs.s3a.path.style.access", "true") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

# Desactivar la generación de código para evitar el límite de los 64 KB de Janino
spark.conf.set("spark.sql.codegen.wholeStage", "false")

# 2. Configuración de MinIO (S3A)
MINIO_ACCESS_KEY = "dpladmin"
MINIO_SECRET_KEY = "dpladmin123"
MINIO_ENDPOINT   = "http://minio:9000"

hconf = spark._jsc.hadoopConfiguration()
hconf.set("fs.s3a.access.key", MINIO_ACCESS_KEY)
hconf.set("fs.s3a.secret.key", MINIO_SECRET_KEY)
hconf.set("fs.s3a.endpoint", MINIO_ENDPOINT)
hconf.set("fs.s3a.path.style.access", "true")
hconf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hconf.set("fs.s3a.connection.ssl.enabled", "false")

# 3. Cargar datos raw
RAW_PATH = "s3a://dpl/raw/speeddating.csv"
df_raw = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .option("sep", ",")
          .csv(RAW_PATH))

# 4. Transformar género a binario y normalizar nulos
df = df_raw.withColumn(
    "gender",
    F.when(F.lower(F.trim(F.col("gender"))) == "male", 1)
     .when(F.lower(F.trim(F.col("gender"))) == "female", 0)
     .otherwise(None)
)

missing_tokens = ["", "NA", "NaN", "nan", "None", "NULL", "null", "?"]
for col_name in df.columns:
    df = df.withColumn(
        col_name,
        F.when(F.col(col_name).isin(missing_tokens), None).otherwise(F.col(col_name))
    )

# 5. Casting de numéricas a float
for c, t in df.dtypes:
    if t == "string":
        df = df.withColumn(c, F.col(c).cast(FloatType()))

# 6. Eliminar columna con exceso de nulos
if "expected_num_interested_in_me" in df.columns:
    df = df.drop("expected_num_interested_in_me")

# 7. Filtrado de rangos y outliers
rangos = {
    "age": (18, 55), "wave": (1, 21), "gender": (0, 1), "age_o": (18, 55),
    "d_age": (0, 37), "samerace": (0, 1), "importance_same_race": (0, 10),
    "importance_same_religion": (0, 10), "pref_o_attractive": (0, 100),
    "pref_o_sincere": (0, 100), "pref_o_intelligence": (0, 100),
    "pref_o_funny": (0, 100), "pref_o_ambitious": (0, 100),
    "pref_o_shared_interests": (0, 100), "attractive_o": (0, 10),
    "sinsere_o": (0, 10), "intelligence_o": (0, 10), "funny_o": (0, 10),
    "ambitous_o": (0, 10), "shared_interests_o": (0, 10),
    "attractive_important": (0, 100), "sincere_important": (0, 100),
    "intellicence_important": (0, 100), "funny_important": (0, 100),
    "ambtition_important": (0, 100), "shared_interests_important": (0, 100),
    "attractive": (0, 10), "sincere": (0, 10), "intelligence": (0, 10),
    "funny": (0, 10), "ambition": (0, 10), "attractive_partner": (0, 10),
    "sincere_partner": (0, 10), "intelligence_partner": (0, 10),
    "funny_partner": (0, 10), "ambition_partner": (0, 10),
    "shared_interests_partner": (0, 10), "sports": (0, 10),
    "tvsports": (0, 10), "exercise": (0, 10), "dining": (0, 10),
    "museums": (0, 10), "art": (0, 10), "hiking": (0, 10), "gaming": (0, 10),
    "clubbing": (0, 10), "reading": (0, 10), "tv": (0, 10), "theater": (0, 10),
    "movies": (0, 10), "concerts": (0, 10), "music": (0, 10),
    "shopping": (0, 10), "yoga": (0, 10), "interests_correlate": (-1, 1),
    "expected_happy_with_sd_people": (0, 10), "expected_num_matches": (0, 20),
    "like": (0, 10), "guess_prob_liked": (0, 10), "met": (0, 1), "match": (0, 1)
}

mask = None
for col, (minv, maxv) in rangos.items():
    if col in df.columns:
        cond = (F.col(col).isNull()) | ((F.col(col) >= minv) & (F.col(col) <= maxv))
        mask = cond if mask is None else (mask & cond)

df = df.filter(mask)

# 7.1. Verificación de suma de columnas (9-14 y 21-26 en notación 1-based de R -> 8:14 y 20:26 en Python)
cols_9_14 = df.columns[8:14]
cols_21_26 = df.columns[20:26]

sum_9_14 = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in cols_9_14)
sum_21_26 = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in cols_21_26)

condicion_9_14 = F.abs(sum_9_14 - 100.0) <= 0.01
condicion_21_26 = F.abs(sum_21_26 - 100.0) <= 0.01

df = df.filter(condicion_9_14 & condicion_21_26)

# 8. Eliminar todas las filas que contengan cualquier valor nulo restante
df_clean = df.dropna()

print("Calculando conteo de filas en el clúster...")
print(f"Filas finales tras eliminar nulos, outliers y restricciones de suma: {df_clean.count()}")

# 9. Guardar en capa curated separada
CURATED_DROPPED = "s3a://dpl/curated_dropped"
print(f"Escribiendo resultado en {CURATED_DROPPED}...")

(df_clean
 .write.mode("overwrite")
 .format("parquet")
 .save(CURATED_DROPPED))

print("¡Escritura OK en:", CURATED_DROPPED)