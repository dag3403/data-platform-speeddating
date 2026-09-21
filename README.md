# Data Platform - Speed Dating Analytics

Plataforma reproducible de ingeniería y analítica de datos construida sobre el
dataset de **Speed Dating**. El proyecto implementa un flujo lakehouse
pequeño, pero completo: ingesta de CSV, almacenamiento de objetos en MinIO,
transformación distribuida con Apache Spark, persistencia relacional en
PostgreSQL, notebooks analíticos y consumo visual mediante Metabase.

> **Estado del repositorio.** Este documento describe el comportamiento que
> implementan actualmente los archivos del proyecto. En particular, los jobs
> ETL no crean hoy una columna `id` con `row_number()` y no hay dashboards de
> Metabase exportados dentro del repositorio. Ambos puntos se explican en las
> secciones correspondientes para distinguir el diseño analítico de la
> implementación efectiva.

## 1. Descripción general

### Objetivo

El sistema permite preparar y analizar las interacciones de citas rápidas para:

- estudiar qué atributos se asocian con que dos participantes hagan *match*;
- comparar dos estrategias de tratamiento de datos faltantes;
- entrenar un clasificador SVM para predecir `match`;
- segmentar los matches mediante clustering;
- publicar datasets y resultados en una base consultable por herramientas BI.

El dataset de entrada es `data/raw/speeddating.csv`. La estructura original
incluye variables demográficas (`age`, `age_o`, `gender`, `wave`), preferencias
(`pref_o_*`, `*_important`), valoraciones de la cita (`attractive`, `funny`,
`intelligence`, etc.), intereses, probabilidades percibidas y las variables
objetivo `met` y `match`.

### Flujo end-to-end

```text
data/raw/speeddating.csv
        |
        | docker compose --profile seed ... (MinIO Client)
        v
MinIO / bucket dpl / raw/
        |
        | Spark + Hadoop S3A
        +--> etl_dropped.py  ------> curated_dropped/  (Parquet)
        |
        +--> etl_imputed.py -------> curated_imputed/  (Parquet)
        |                                  |
        |                                  +--> train_svm.py
        |                                  |       |
        |                                  |       +--> metrics/ (JSON/text)
        |                                  |
        |                                  +--> load_minio_to_postgres.py
        v
PostgreSQL (tablas public.*)
        |
        v
Metabase: preguntas, modelos y dashboards
```

MinIO expone una API compatible con S3. Spark accede a ella mediante `s3a://`
y el conector `hadoop-aws`; por tanto, los contenedores se comunican usando los
nombres DNS internos `minio`, `spark-master` y `postgres`, no `localhost`.

## 2. Arquitectura de contenedores

Todos los servicios se definen en [docker-compose.yml](./docker-compose.yml) y
se conectan a la red interna `dpl_net`. Los datos persistentes se guardan en
volúmenes Docker nombrados para sobrevivir a la recreación de contenedores.

### Servicios

| Servicio | Imagen | Puertos publicados | Responsabilidad |
|---|---|---:|---|
| `minio` | `${MINIO_IMAGE}` (`quay.io/minio/minio:latest`) | `9000` API, `9001` consola | Object storage S3 para raw, curated y métricas |
| `mc` | `${MC_IMAGE}` (`minio/mc:latest`) | Ninguno | Inicialización opcional del bucket y copia de `./data` |
| `spark-master` | `${SPARK_IMAGE}` (`bitnamilegacy/spark:3.5.0`) | `7077` Spark master, `8080` UI | Coordinación de aplicaciones Spark |
| `spark-worker-1` | `${SPARK_IMAGE}` | Ninguno | Ejecutor Spark, 2 cores y 4 GB |
| `spark-worker-2` | `${SPARK_IMAGE}` | Ninguno | Ejecutor Spark, 2 cores y 4 GB |
| `jupyter` | Imagen construida desde `Dockerfile.jupyter` | `8888` Jupyter | Notebook server y punto práctico para lanzar PySpark |
| `postgres` | `${POSTGRES_IMAGE}` (`postgres:15`) | `5432` | Almacén relacional para consumo SQL/BI |
| `metabase` | `${METABASE_IMAGE}` (`metabase/metabase:v0.49.12`) | `3000` | Exploración, preguntas y dashboards |

### Profiles de Compose

- `base`: MinIO, PostgreSQL, master/worker de Spark y Jupyter.
- `seed`: MinIO y `mc`; el cliente copia `./data/raw` a `s3://dpl/raw/`.
- `viz`: PostgreSQL y Metabase.

Los servicios con `depends_on` expresan el orden de arranque, pero no
constituyen una comprobación de salud. Si un job se lanza demasiado pronto,
espere a que la UI o los logs indiquen que el servicio está listo.

### Volúmenes y configuración

- `minio_data:/data`: objetos persistentes de MinIO.
- `postgres_data:/var/lib/postgresql/data`: catálogo y tablas de PostgreSQL.
- `metabase_data:/metabase.db`: estado de Metabase.
- `./data:/seed:ro` en `mc`: dataset local, montado solo para lectura.
- `./spark/jobs:/opt/bitnami/spark/jobs:ro` en el master.
- `./notebooks:/home/jovyan/work` y `./spark/jobs:/home/jovyan/jobs:ro` en
  Jupyter.
- `./postgres/initdb:/docker-entrypoint-initdb.d` en PostgreSQL. Actualmente
  el directorio no contiene scripts SQL versionados; el esquema se crea
  implícitamente al escribir tablas con JDBC.

El archivo `.env` parametriza imágenes, credenciales, nombres y el token de
Jupyter. No se deben reutilizar sus credenciales de ejemplo en un entorno
expuesto; cambie `MINIO_ROOT_PASSWORD`, `POSTGRES_PASSWORD` y
`JUPYTER_TOKEN`, y mantenga `.env` fuera de control de versiones cuando
contenga secretos reales.

## 3. Estructura del proyecto

```text
.
├── app.py
├── artifacts/
│   └── svm_inference_bundle.joblib
├── data/
│   └── raw/
│       └── speeddating.csv
├── ml_service/
│   ├── __init__.py
│   └── production_pipeline.py
├── notebooks/
│   ├── speeddating.ipynb
│   └── análisis_svm.ipynb
├── spark/
│   └── jobs/
│       ├── etl_dropped.py
│       ├── etl_imputed.py
│       ├── load_minio_to_postgres.py
│       └── train_svm.py
├── Dockerfile.api
├── Dockerfile.jupyter
├── requirements.txt
├── docker-compose.yml
├── .env
└── .gitignore
```

El `.gitignore` excluye cachés de Python, entornos virtuales, volúmenes locales
y el contenido del directorio raw (salvo `.gitkeep`). Esto evita versionar
artefactos pesados y datos potencialmente sensibles.

## 4. Inferencia en producción con FastAPI

El proyecto ahora incluye un servicio de inferencia que reutiliza exactamente la
misma lógica de preprocessing que se emplea durante el entrenamiento. El flujo
es:

```text
raw CSV
  -> apply_etl_preprocessing()  (sin fit en producción)
  -> IterativeImputer.fit(...) en entrenamiento
  -> LogisticRegressionCV para LASSO
  -> SVM RBF
  -> artifacts/svm_inference_bundle.joblib
  -> FastAPI /predict
```

La etapa `apply_etl_preprocessing()` conserva la estrategia existente de:
- normalización de `gender`;
- reemplazo de tokens nulos;
- cast de string a float;
- eliminación de `expected_num_interested_in_me`;
- validación de rangos y sumas parciales;
- imputación iterativa con `IterativeImputer` y los parámetros ya definidos.

Durante la inferencia, el bundle serializado guarda el `imputer` ajustado y la
lista de columnas dummy/seleccionadas para hacer únicamente `transform`, nunca
`fit` ni `fit_transform` sobre producción. El servicio corre con Docker mediante
`Dockerfile.api` y se exposa en el puerto `8000`.

## 5. ETL con Spark

Los jobs leen el CSV desde `s3a://dpl/raw/speeddating.csv`, configuran el
endpoint S3A de MinIO (`http://minio:9000`) y escriben Parquet en el bucket
`dpl`. Ambos ETL desactivan `spark.sql.codegen.wholeStage` para evitar el
límite de generación de código de Janino observado con este esquema ancho.

### `etl_dropped.py`

[etl_dropped.py](./spark/jobs/etl_dropped.py) implementa la rama de limpieza
por eliminación:

1. Inicializa una `SparkSession` con `hadoop-aws:3.3.4` y
   `aws-java-sdk-bundle:1.12.262`.
2. Lee el CSV con cabecera, inferencia de esquema y separador coma.
3. Normaliza `gender`: `male` se convierte en `1`, `female` en `0` y otros
   valores en nulo.
4. Convierte tokens vacíos o equivalentes (`NA`, `NaN`, `None`, `NULL`, `?`,
   entre otros) en `NULL`.
5. Hace cast a `float` de las columnas originalmente string.
6. Elimina `expected_num_interested_in_me`, que presenta una proporción muy
   elevada de faltantes.
7. Filtra rangos válidos para edad, wave, escalas de valoración,
   preferencias, intereses, `met` y `match`. Los nulos se toleran durante este
   filtro para que la eliminación final sea explícita.
8. Verifica que los grupos de columnas de preferencias sumen aproximadamente
   `100` (tolerancia `0.01`).
9. Ejecuta `dropna()` para eliminar cualquier fila con nulos restantes.
10. Sobrescribe `s3a://dpl/curated_dropped` en formato Parquet.

### `etl_imputed.py`

[etl_imputed.py](./spark/jobs/etl_imputed.py) comparte la normalización,
validación de rangos y restricciones de suma de la rama anterior, pero conserva
las filas con faltantes. No ajusta ningún imputador: materializa la limpieza
estructural y escribe el Parquet en `s3a://dpl/curated_imputed`.

La imputación iterativa se ajusta en
[train_svm.py](./spark/jobs/train_svm.py), después de separar train/test. Así
los parámetros de `IterativeImputer` se aprenden únicamente con `X_train` y se
reutilizan mediante `transform` sobre test y producción.

### Sobre la columna relacional `id`

Una clave técnica reproducible podría generarse en Spark con una ventana,
por ejemplo:

```python
from pyspark.sql.window import Window

window = Window.orderBy("wave", "age", "age_o", "match")
df = df.withColumn("id", F.row_number().over(window))
```

Sin embargo, **esa lógica no existe actualmente** en `etl_dropped.py`,
`etl_imputed.py`, `train_svm.py` ni en el cargador. Tampoco se encuentra
`row_number()` ni una columna `id` en los artefactos del repositorio. Si se
incorpora, el orden debe usar columnas que definan una regla estable y se debe
validar la unicidad; un orden incompleto puede producir identificadores
distintos entre ejecuciones. El cargador conserva las columnas que recibe y no
inventa una clave.

## 5. Carga de MinIO a PostgreSQL

[load_minio_to_postgres.py](./spark/jobs/load_minio_to_postgres.py) es un job
genérico de ingestión:

- recibe bucket, prefijo, endpoint y credenciales de MinIO, además de host,
  puerto, base, usuario, contraseña y esquema PostgreSQL mediante argumentos o
  variables de entorno;
- recorre recursivamente el prefijo S3 y descubre archivos `.csv` y directorios
  Parquet;
- deriva nombres de tabla a partir del archivo o directorio, reemplazando
  caracteres no alfanuméricos por `_`, en minúsculas;
- lee CSV con cabecera e inferencia de esquema, o Parquet directamente;
- normaliza nombres de columnas, evita colisiones añadiendo sufijos, convierte
  tokens vacíos en nulos y aplica `dropDuplicates()`;
- escribe cada DataFrame mediante JDBC en modo `overwrite`, con
  `batchsize=1000`, `numPartitions=4`, `truncate=true` y
  `org.postgresql.Driver`.

El job declara `hadoop-aws:3.3.4`, `aws-java-sdk-bundle:1.12.262` y
`postgresql:42.7.3` mediante `spark.jars.packages`. Además, fija el JAR de
PostgreSQL en `SPARK_HOME/jars` con `spark.files` y `spark.jars`, lo que evita
que los executors carezcan del driver JDBC.

Ejemplo de rutas resultantes (dependen del contenido de MinIO):

```text
s3a://dpl/raw/speeddating.csv       -> public.speeddating
s3a://dpl/curated_dropped/          -> public.curated_dropped
s3a://dpl/curated_imputed/          -> public.curated_imputed
```

El descubrimiento real prevalece sobre estos ejemplos y puede crear una tabla
por cada dataset encontrado bajo el prefijo indicado.

## 6. Análisis y modelado

### `notebooks/speeddating.ipynb`

Este notebook documenta el análisis exploratorio y la preparación inicial:

- crea una sesión Spark conectada a MinIO;
- inspecciona el esquema y los faltantes;
- identifica el alto porcentaje de nulos de
  `expected_num_interested_in_me`;
- explora las validaciones de rangos y las restricciones de suma;
- genera una salida Parquet de análisis bajo `s3a://dpl/curated` en el flujo
  del notebook.

El notebook es una bitácora exploratoria y no reemplaza a los jobs ETL
versionados; sus transformaciones deben compararse con las variantes
`curated_dropped` y `curated_imputed` antes de usarse como fuente oficial.

### `notebooks/análisis_svm.ipynb`

Este notebook consume las métricas generadas en MinIO y desarrolla el análisis
de resultados:

- compara `dropped` e `imputed`;
- revisa accuracy, sensibilidad, especificidad, AUC y matriz de confusión;
- realiza un análisis de clúster sobre registros con `match`;
- usa estandarización, PCA para visualización, método del codo/WSS y
  `KMeans`;
- visualiza la dispersión coloreada por clúster;
- calcula medias/centroides por clúster para interpretar perfiles.

Los resultados guardados en el notebook muestran, para la ejecución registrada,
un AUC de aproximadamente `0.8373` para `dropped` y `0.8545` para `imputed`.
Estos números son salidas históricas del notebook, no una garantía para una
ejecución con datos o dependencias diferentes.

### `train_svm.py`

[train_svm.py](./spark/jobs/train_svm.py) permite seleccionar la fuente con
`--dataset dropped|imputed` y es la única fuente de verdad para crear el bundle
`artifacts/svm_inference_bundle.joblib`:

1. Lee el Parquet curado desde MinIO y lo convierte a Pandas.
2. Usa `match` como variable binaria objetivo.
3. Divide primero en entrenamiento/test `70/30`, con `random_state=456` y
   estratificación por `match`.
4. Ajusta `IterativeImputer(max_iter=10, random_state=42)` únicamente sobre
   `X_train`; transforma `X_train` y `X_test`.
5. Ajusta `pd.get_dummies` y guarda `dummy_columns` para alinear exactamente
   futuras entradas.
6. Ajusta `LogisticRegressionCV` con penalización L1 (LASSO), validación
   cruzada de 10 particiones y scoring ROC-AUC únicamente sobre training.
7. Balancea únicamente el training mediante `RandomOverSampler`.
8. Entrena `SVC(kernel="rbf", probability=True, random_state=42)`.
9. Evalúa sobre `X_test` sin oversampling con accuracy, sensibilidad,
   especificidad, AUC y matriz de confusión.
10. Serializa el modelo, el imputador, `dummy_columns`, `selected_features` y
    las métricas en el bundle. También escribe las métricas en la ruta
   `s3a://dpl/metrics/`. La implementación escribe el JSON como texto en
   `svm_output_temp_<dataset>`; el nombre `svm_metrics_<dataset>.json` aparece
   como ruta declarada, pero no se utiliza en la operación final de escritura.

El paso Spark-to-Pandas también impone una limitación de memoria para datasets
grandes. Las dependencias de modelado (`scikit-learn`, `imbalanced-learn` y
`pandas`) deben estar disponibles en el entorno desde el que se ejecute el
job.

## 7. Visualización y BI con Metabase

Metabase se configura para usar PostgreSQL como su propia base de aplicación:

- host interno: `postgres`;
- puerto: `5432`;
- base: valor de `POSTGRES_DB`;
- usuario y contraseña: `POSTGRES_USER`/`POSTGRES_PASSWORD`.

Después de cargar datasets, se añade en Metabase una conexión de datos a
PostgreSQL con los mismos valores internos. En la primera sincronización,
Metabase inspecciona el esquema `public`, tipos y tablas disponibles. Si una
tabla fue reemplazada por JDBC, conviene ejecutar **Admin > Databases >
Sync database schema now** y revisar los metadatos antes de construir
preguntas.

Vistas recomendadas:

- **Dispersión de atributos:** `attractive_partner` frente a
  `shared_interests_partner`, coloreada por `match`.
- **Comparación de medias:** promedios de `like`, `funny_partner`,
  `intelligence_partner` y `guess_prob_liked`, agrupados por `match` o
  `wave`.
- **Clústeres:** si se persisten las etiquetas calculadas por el notebook,
  conteo de registros y medias por `cluster`.
- **Calidad de datos:** recuentos por dataset, nulos y distribución de
  `match`.
- **Modelo:** tarjetas con accuracy, sensibilidad, especificidad y AUC
  leídas desde una tabla o JSON previamente cargado.

El repositorio no contiene una exportación `.json` de preguntas ni un dashboard
de Metabase versionado. Por ello, los nombres y disposición finales de los
paneles dependen de la configuración realizada en la instancia local.

## 8. Guía de despliegue y uso

### Requisitos

- Docker Desktop con Docker Compose v2.
- Al menos 8 GB de memoria asignable a Docker; Spark usa dos workers de 4 GB.
- Puertos libres `3000`, `5432`, `7077`, `8080`, `8888`, `9000` y `9001`.
- El CSV disponible en `data/raw/speeddating.csv`.

### 8.1 Levantar la plataforma

Desde la raíz del repositorio:

```powershell
docker compose --profile base --profile viz up -d
```

Construya/arranque Jupyter y compruebe el estado:

```powershell
docker compose ps
docker compose logs -f spark-master
```

Interfaces:

- Jupyter: <http://localhost:8888> (token configurado en `JUPYTER_TOKEN`).
- MinIO Console: <http://localhost:9001>.
- Spark Master UI: <http://localhost:8080>.
- Metabase: <http://localhost:3000>.
- PostgreSQL: `localhost:5432` desde el host; `postgres:5432` desde Compose.

### 8.2 Sembrar el bucket raw

Ejecute el cliente MinIO después de que MinIO esté disponible:

```powershell
docker compose --profile seed run --rm mc
```

La tarea crea (o reutiliza) el bucket `dpl` y copia `./data/raw/` a
`dpl/raw/`. Verifique el objeto desde la consola o con:

```powershell
docker compose run --rm mc ls dpl/raw
```

### 8.3 Ejecutar ETL

Los jobs están montados en el master y en Jupyter. Una forma reproducible de
ejecutarlos es usar `spark-submit` dentro del contenedor master:

```powershell
docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/etl_dropped.py

docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/etl_imputed.py
```

Confirme que ambas rutas Parquet existan en MinIO antes de continuar. Los jobs
usan credenciales configuradas en el código y en Compose; en un despliegue real
deben unificarse mediante variables de entorno o un gestor de secretos.

### 8.4 Entrenar y guardar métricas

```powershell
docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/train_svm.py --dataset dropped

docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/train_svm.py --dataset imputed
```

Revise los logs para las métricas y la consola de MinIO para los objetos bajo
`metrics/`.

### 8.5 Cargar a PostgreSQL

Para cargar todos los CSV/Parquet del bucket:

```powershell
docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/load_minio_to_postgres.py `
  --bucket dpl `
  --prefix ""
```

Para limitar la carga a un prefijo:

```powershell
docker compose exec spark-master spark-submit `
  --master spark://spark-master:7077 `
  /opt/bitnami/spark/jobs/load_minio_to_postgres.py `
  --bucket dpl `
  --prefix curated_imputed
```

Como el modo de escritura es `overwrite`, una nueva ejecución reemplaza las
tablas descubiertas. Esto es apropiado para una carga reproducible, pero debe
revisarse antes de usarlo como proceso incremental.

### 8.6 Configurar Metabase

1. Abra <http://localhost:3000> y complete el asistente inicial.
2. Añada una base PostgreSQL de tipo **PostgreSQL**.
3. Use `postgres` como host, `5432` como puerto y los valores de `.env` para
   base, usuario y contraseña.
4. Sincronice el esquema y confirme las tablas en `public`.
5. Cree preguntas, gráficos y un dashboard con las vistas sugeridas.

### 8.7 Apagado y limpieza

Para detener sin borrar datos:

```powershell
docker compose down
```

Para borrar también los volúmenes persistentes (acción destructiva sobre
MinIO, PostgreSQL y Metabase):

```powershell
docker compose down -v
```

## 9. Operación, limitaciones y siguientes pasos

- Externalizar todas las credenciales de los scripts ETL, que actualmente
  contienen valores por defecto codificados.
- Añadir healthchecks y reintentos para que `depends_on` no dependa solo del
  orden de creación.
- Versionar un esquema SQL o migraciones si se requiere un modelo relacional
  estable.
- Definir y probar una clave `id` determinista antes de usar relaciones entre
  tablas; hoy no existe una clave generada por ventana.
- Persistir las etiquetas de clúster y las métricas en tablas PostgreSQL si
  Metabase debe consumirlas de forma nativa.
- Sustituir `toPandas()` por transformaciones distribuidas para escalar.
- Exportar la configuración de Metabase y añadirla al control de versiones si
  se necesita reproducibilidad completa de BI.
- Fijar versiones de imágenes que actualmente usan `latest`, especialmente
  MinIO y `mc`, para despliegues reproducibles.
