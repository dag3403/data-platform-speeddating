from pyspark.sql import SparkSession

def main():
    # 1. Inicializar la sesión de Spark configurada para MinIO (S3)
    spark = SparkSession.builder \
        .appName("SpeedDatingETL") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "dpladmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "dpladmin123") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .getOrCreate()

    print("Sesión de Spark iniciada correctamente.")

    # 2. Leer el CSV crudo desde MinIO (cambia el nombre del archivo si es distinto al tuyo)
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("s3a://dpl/raw/speeddating.csv")

    print(f"Número de filas cargadas: {df.count()}")
    print(f"Número de columnas: {len(df.columns)}")

    # Mostramos las primeras filas para verificar
    df.show(5)

    spark.stop()

if __name__ == "__main__":
    main()