"""Load CSV and Snappy Parquet datasets from MinIO into PostgreSQL.

The job discovers datasets below an S3 bucket/prefix. A CSV file becomes a
table named after its file, while a Parquet directory becomes a table named
after the directory containing its part files.
"""

import argparse
import os
import re
from collections import OrderedDict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load MinIO CSV/Parquet datasets into PostgreSQL."
    )
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET", "dpl"))
    parser.add_argument("--prefix", default=os.getenv("MINIO_PREFIX", ""))
    parser.add_argument("--minio-endpoint", default=os.getenv("MINIO_ENDPOINT", "http://minio:9000"))
    parser.add_argument(
        "--minio-access-key",
        default=os.getenv("MINIO_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID")),
    )
    parser.add_argument(
        "--minio-secret-key",
        default=os.getenv("MINIO_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY")),
    )
    parser.add_argument("--postgres-host", default=os.getenv("POSTGRES_HOST", "postgres"))
    parser.add_argument("--postgres-port", default=os.getenv("POSTGRES_PORT", "5432"))
    parser.add_argument("--postgres-database", default=os.getenv("POSTGRES_DB", "dpl"))
    parser.add_argument("--postgres-user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument("--postgres-password", default=os.getenv("POSTGRES_PASSWORD", "postgres"))
    parser.add_argument("--postgres-schema", default=os.getenv("POSTGRES_SCHEMA", "public"))
    return parser.parse_args()


def configure_spark(args: argparse.Namespace) -> SparkSession:
    return (
        SparkSession.builder.appName("MinIO to PostgreSQL loader")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
            "org.postgresql:postgresql:42.7.3",
        )
        .config("spark.files", "/usr/local/spark-3.5.0-bin-hadoop3/jars/postgresql-42.7.3.jar")
        .config("spark.jars", "/usr/local/spark-3.5.0-bin-hadoop3/jars/postgresql-42.7.3.jar")
        .config("spark.hadoop.fs.s3a.endpoint", args.minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", args.minio_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", args.minio_secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )


def discover_datasets(spark: SparkSession, root: str) -> OrderedDict[str, str]:
    """Return one source path per CSV file or Parquet directory."""
    path_type = spark._jvm.org.apache.hadoop.fs.Path
    filesystem = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        path_type(root).toUri(), spark._jsc.hadoopConfiguration()
    )
    datasets: OrderedDict[str, str] = OrderedDict()

    def visit(path: str) -> None:
        for status in filesystem.listStatus(path_type(path)):
            current = status.getPath().toString()
            name = status.getPath().getName()
            if status.isDirectory():
                visit(current)
            elif name.lower().endswith(".csv"):
                table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", name[:-4]).strip("_").lower()
                datasets[table_name] = current
            elif name.lower().endswith((".parquet", ".snappy.parquet")):
                parent = status.getPath().getParent().getName()
                table_name = re.sub(r"[^a-zA-Z0-9_]+", "_", parent).strip("_").lower()
                datasets[table_name] = status.getPath().getParent().toString()

    visit(root.rstrip("/"))
    return datasets


def normalize_dataframe(df: DataFrame) -> DataFrame:
    """Normalize names and common missing values without changing data types."""
    renamed = df
    used_names: set[str] = set()
    for column in df.columns:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", column.strip()).strip("_").lower()
        name = name or "column"
        base_name = name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        renamed = renamed.withColumnRenamed(column, name)

    for column, data_type in renamed.dtypes:
        if data_type == "string":
            renamed = renamed.withColumn(
                column,
                F.when(
                    F.trim(F.col(column)).isin("", "NA", "NaN", "nan", "NULL", "null", "?"),
                    F.lit(None),
                ).otherwise(F.trim(F.col(column))),
            )
    return renamed.dropDuplicates()


def load_dataset(spark: SparkSession, path: str) -> DataFrame:
    if path.lower().endswith(".csv"):
        return spark.read.option("header", True).option("inferSchema", True).csv(path)
    return spark.read.parquet(path)


def write_table(df: DataFrame, table_name: str, args: argparse.Namespace) -> None:
    jdbc_url = (
        f"jdbc:postgresql://{args.postgres_host}:{args.postgres_port}/"
        f"{args.postgres_database}"
    )
    table = f'"{args.postgres_schema}"."{table_name}"'
    (
        df.write.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", args.postgres_user)
        .option("password", args.postgres_password)
        .option("driver", "org.postgresql.Driver")
        .option("batchsize", "1000")
        .option("numPartitions", "4")
        .option("truncate", "true")
        .mode("overwrite")
        .save()
    )


def main() -> None:
    args = parse_args()
    spark = configure_spark(args)
    spark.sparkContext.setLogLevel("WARN")
    root = f"s3a://{args.bucket}/{args.prefix.strip('/')}".rstrip("/")

    try:
        datasets = discover_datasets(spark, root)
        if not datasets:
            raise RuntimeError(f"No CSV or Parquet files found below {root}")

        print(f"Datasets discovered: {', '.join(datasets)}")
        for table_name, path in datasets.items():
            print(f"Loading {path} into {args.postgres_schema}.{table_name}")
            dataframe = normalize_dataframe(load_dataset(spark, path))
            row_count = dataframe.count()
            write_table(dataframe, table_name, args)
            print(f"Loaded {args.postgres_schema}.{table_name}: {row_count} rows")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()