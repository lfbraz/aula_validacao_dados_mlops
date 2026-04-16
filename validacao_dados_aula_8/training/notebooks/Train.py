# Databricks notebook source
##################################################################################
# Notebook de Treinamento do Modelo
#
# Este notebook treina um modelo de regressão LightGBM para prever o valor de
# corridas de táxi (fare_amount) a partir de dados sintéticos no Unity Catalog.
#
# Pipeline:
#   1. Carrega dados de treino do Delta Lake (Unity Catalog)
#   2. Realiza feature engineering simples
#   3. Treina um modelo LightGBM com autolog do MLflow
#   4. Registra o modelo no Unity Catalog Model Registry
#   5. Retorna o URI do modelo para as próximas etapas do workflow
#
# Configuração no bundle (resources/model-workflow-resource.yml):
#   - Faz parte do job "model_training_job" como task "Train"
#   - O model_uri gerado é passado como task value para as tasks seguintes
#
# Parâmetros:
#   * env                 - Ambiente de execução (staging ou prod). Padrão: "staging"
#   * training_data_path  - Tabela Delta de treino (three-level UC name)
#   * experiment_name     - Nome do experimento MLflow
#   * model_name          - Nome completo do modelo no Unity Catalog
##################################################################################

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt

# COMMAND ----------

# DBTITLE 1, Parâmetros do notebook
dbutils.widgets.dropdown("env", "staging", ["staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# Tabela de treino no Unity Catalog (criada pelo DataSetup.py)
dbutils.widgets.text(
    "training_data_path",
    "dev.validacao_dados_aula_8.taxi_fares_train",
    label="Tabela de treino (three-level UC name)",
)

# Experimento MLflow — será criado automaticamente se não existir
dbutils.widgets.text(
    "experiment_name",
    f"/dev-validacao_dados_aula_8-experiment",
    label="Nome do experimento MLflow",
)

# Nome do modelo no Unity Catalog Model Registry
dbutils.widgets.text(
    "model_name",
    "dev.validacao_dados_aula_8.validacao_dados_aula_8-model",
    label="Nome completo do modelo (three-level UC name)",
)

# COMMAND ----------

# DBTITLE 1, Definir variáveis
input_table_path = dbutils.widgets.get("training_data_path")
experiment_name  = dbutils.widgets.get("experiment_name")
model_name       = dbutils.widgets.get("model_name")

print(f"Ambiente       : {env}")
print(f"Tabela treino  : {input_table_path}")
print(f"Experimento    : {experiment_name}")
print(f"Modelo UC      : {model_name}")

# COMMAND ----------

# DBTITLE 1, Configurar MLflow
import mlflow

mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri("databricks-uc")  # Usar Unity Catalog como model registry

print(f"✅ Experimento MLflow configurado: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregar Dados de Treino
# MAGIC
# MAGIC Os dados foram criados pelo notebook `DataSetup.py` e estão armazenados como
# MAGIC Delta Table no Unity Catalog. Carregamos diretamente com Spark.

# COMMAND ----------

# DBTITLE 1, Carregar dados do Delta Lake
training_df = spark.table(input_table_path)

print(f"✅ Dados carregados: {training_df.count():,} registros, {len(training_df.columns)} colunas")
print(f"\nSchema:")
training_df.printSchema()
display(training_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering
# MAGIC
# MAGIC Criamos algumas features derivadas para melhorar o modelo:
# MAGIC - `is_rush_hour`: flag para horário de pico
# MAGIC - `is_weekend`: flag para final de semana
# MAGIC - `is_bad_weather`: flag para mau tempo
# MAGIC - `distance_x_passengers`: interação entre distância e número de passageiros

# COMMAND ----------

# DBTITLE 1, Feature engineering
from pyspark.sql import functions as F

training_df = (
    training_df
    # Horário de pico: 7–9h ou 17–19h
    .withColumn(
        "is_rush_hour",
        F.when(
            ((F.col("hour_of_day") >= 7) & (F.col("hour_of_day") <= 9)) |
            ((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 19)),
            1
        ).otherwise(0)
    )
    # Final de semana
    .withColumn(
        "is_weekend",
        F.when(F.col("day_of_week") >= 5, 1).otherwise(0)
    )
    # Mau tempo (score < 3)
    .withColumn(
        "is_bad_weather",
        F.when(F.col("weather_score") < 3.0, 1).otherwise(0)
    )
    # Feature de interação: distância × passageiros
    .withColumn(
        "distance_x_passengers",
        F.col("trip_distance") * F.col("num_passengers")
    )
)

print(f"✅ Features criadas: {training_df.columns}")
display(training_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Treinar Modelo LightGBM
# MAGIC
# MAGIC Utilizamos o LightGBM, um algoritmo de gradient boosting eficiente para
# MAGIC dados tabulares. O MLflow autolog captura automaticamente:
# MAGIC - Hiperparâmetros do modelo
# MAGIC - Métricas de avaliação (RMSE, MAE, R²)
# MAGIC - O artefato do modelo treinado

# COMMAND ----------

# DBTITLE 1, Preparar dados para sklearn/LightGBM
import mlflow
import lightgbm as lgb
import mlflow.lightgbm
from sklearn.model_selection import train_test_split

# Colunas que NÃO são features (target + colunas derivadas que não entram no modelo)
TARGET_COL = "fare_amount"

# Todas as colunas numéricas (incluindo as features engineered)
feature_cols = [c for c in training_df.columns if c != TARGET_COL]

# Converter para Pandas para treinamento
data = training_df.toPandas()
X    = data[feature_cols]
y    = data[TARGET_COL]

# Split treino/teste local (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Features usadas ({len(feature_cols)}): {feature_cols}")
print(f"  Treino : {X_train.shape}")
print(f"  Teste  : {X_test.shape}")

# COMMAND ----------

# DBTITLE 1, Treinar com MLflow autolog
mlflow.lightgbm.autolog()

# Criar datasets LightGBM
train_lgb = lgb.Dataset(X_train, label=y_train.values)
test_lgb  = lgb.Dataset(X_test,  label=y_test.values)

# Hiperparâmetros do modelo
# Estes valores são simples para fins didáticos — em produção, use hyperparameter tuning
params = {
    "objective"  : "regression",
    "metric"     : ["rmse", "mae"],
    "num_leaves" : 31,
    "learning_rate": 0.05,
    "n_estimators" : 200,
    "verbose"    : -1,
}

print("🚀 Iniciando treinamento LightGBM com MLflow autolog...")
model = lgb.train(
    params,
    train_lgb,
    num_boost_round=200,
    valid_sets=[test_lgb],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
)

print("✅ Treinamento concluído!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Registrar Modelo no Unity Catalog
# MAGIC
# MAGIC O modelo treinado é registrado no Unity Catalog Model Registry. Isso permite
# MAGIC versionamento, governança e auditoria centralizada do modelo.

# COMMAND ----------

# DBTITLE 1, Log e registro do modelo
from mlflow.tracking import MlflowClient

# Exemplo de entrada para inferência (schema do modelo)
input_example = X_train.iloc[[0]]

# Registrar modelo no Unity Catalog
mlflow.lightgbm.log_model(
    model,
    artifact_path="lgb_model",
    input_example=input_example,
    registered_model_name=model_name,
)

print(f"✅ Modelo registrado no Unity Catalog: {model_name}")

# COMMAND ----------

# DBTITLE 1, Obter versão do modelo registrado
def get_latest_model_version(model_name: str) -> int:
    """Retorna a versão mais recente do modelo no Unity Catalog."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return 1
    return max(int(mv.version) for mv in versions)

model_version = get_latest_model_version(model_name)
model_uri     = f"models:/{model_name}/{model_version}"

print(f"✅ Modelo registrado como versão: {model_version}")
print(f"   URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Passar Resultados para as Próximas Tasks
# MAGIC
# MAGIC O `model_uri` é passado como **task value** para as tasks seguintes no workflow
# MAGIC (ModelValidation → ModelDeployment). Esta é a forma nativa do Databricks de
# MAGIC passar dados entre tasks de um mesmo job.

# COMMAND ----------

# DBTITLE 1, Publicar task values para o workflow
dbutils.jobs.taskValues.set("model_uri",     model_uri)
dbutils.jobs.taskValues.set("model_name",    model_name)
dbutils.jobs.taskValues.set("model_version", model_version)

print(f"✅ Task values publicados:")
print(f"   model_uri     = {model_uri}")
print(f"   model_name    = {model_name}")
print(f"   model_version = {model_version}")

dbutils.notebook.exit(model_uri)
