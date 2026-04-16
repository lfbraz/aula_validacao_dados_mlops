# Databricks notebook source
##################################################################################
# Notebook de Setup: Criação de Tabelas com Dados Sintéticos
#
# Este notebook cria as tabelas de exemplo no Unity Catalog com dados sintéticos
# para demonstração do pipeline de MLOps no contexto da aula.
#
# O cenário simulado é a previsão do valor de corridas de táxi (regressão),
# criando um dataset simples e interpretável para fins didáticos.
#
# Tabelas criadas:
#   - <catalog>.<schema>.taxi_fares_train  → dados de treino
#   - <catalog>.<schema>.taxi_fares_val    → dados de validação
#   - <catalog>.<schema>.taxi_scoring      → dados para inferência em batch
#
# Pré-requisitos:
#   - Catalog e schema já criados no Unity Catalog
#   - Permissões: CREATE TABLE, MODIFY no schema
#
# Parâmetros:
#   * catalog_name  - Nome do catalog Unity Catalog (ex: dev, staging, prod)
#   * schema_name   - Nome do schema (padrão: mlops_demo)
##################################################################################

# COMMAND ----------

# DBTITLE 1, Parâmetros do notebook
dbutils.widgets.text("catalog_name", "dev", "Catalog Name")
dbutils.widgets.text("schema_name", "validacao_dados_aula_8", "Schema Name")
dbutils.widgets.text("n_train_samples", "5000", "Número de registros de treino")
dbutils.widgets.text("n_scoring_samples", "500", "Número de registros para scoring")
dbutils.widgets.text("random_seed", "42", "Seed aleatório (reprodutibilidade)")

# COMMAND ----------

catalog_name    = dbutils.widgets.get("catalog_name")
schema_name     = dbutils.widgets.get("schema_name")
n_train         = int(dbutils.widgets.get("n_train_samples"))
n_scoring       = int(dbutils.widgets.get("n_scoring_samples"))
random_seed     = int(dbutils.widgets.get("random_seed"))

print(f"Catalog  : {catalog_name}")
print(f"Schema   : {schema_name}")
print(f"Treino   : {n_train} registros")
print(f"Scoring  : {n_scoring} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração do Catalog e Schema
# MAGIC
# MAGIC Garantimos que o catalog e o schema existem antes de criar as tabelas.

# COMMAND ----------

# DBTITLE 1, Criar schema (se não existir)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
print(f"✅ Schema '{catalog_name}.{schema_name}' pronto.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Geração dos Dados Sintéticos
# MAGIC
# MAGIC Geramos um dataset de corridas de táxi com as seguintes features:
# MAGIC
# MAGIC | Coluna            | Tipo    | Descrição                                               |
# MAGIC |-------------------|---------|---------------------------------------------------------|
# MAGIC | `trip_distance`   | float   | Distância da corrida em km (0.5 a 30 km)               |
# MAGIC | `num_passengers`  | int     | Número de passageiros (1 a 6)                          |
# MAGIC | `hour_of_day`     | int     | Hora de início da corrida (0 a 23)                     |
# MAGIC | `day_of_week`     | int     | Dia da semana (0 = segunda, 6 = domingo)               |
# MAGIC | `weather_score`   | float   | Índice climático (0 = péssimo, 10 = ótimo)             |
# MAGIC | `pickup_zone`     | int     | Zona de origem (1 a 5)                                 |
# MAGIC | `dropoff_zone`    | int     | Zona de destino (1 a 5)                                |
# MAGIC | `fare_amount`     | float   | **TARGET**: valor da corrida em USD                    |
# MAGIC
# MAGIC A fórmula base para o valor da corrida é:
# MAGIC ```
# MAGIC fare = 2.50 (base)
# MAGIC      + trip_distance × 1.75
# MAGIC      + num_passengers × 0.20
# MAGIC      + rush_hour_bonus (0 ou 2.50)
# MAGIC      + weekend_bonus (0 ou 1.00)
# MAGIC      + weather_penalty (até +3.00 em mau tempo)
# MAGIC      + ruído gaussiano
# MAGIC ```

# COMMAND ----------

# DBTITLE 1, Gerar dados sintéticos com NumPy/Pandas
import numpy as np
import pandas as pd

rng = np.random.default_rng(random_seed)

def generate_taxi_data(n: int, seed: int) -> pd.DataFrame:
    """
    Gera um DataFrame Pandas com dados sintéticos de corridas de táxi.
    O valor da corrida (fare_amount) é determinístico a partir das features
    + um pequeno ruído gaussiano, para facilitar o aprendizado pelo modelo.
    """
    rng_local = np.random.default_rng(seed)

    # --- Features ---
    trip_distance  = rng_local.uniform(0.5, 30.0, n).round(2)      # km
    num_passengers = rng_local.integers(1, 7, n)                    # 1–6
    hour_of_day    = rng_local.integers(0, 24, n)                   # 0–23
    day_of_week    = rng_local.integers(0, 7, n)                    # 0–6
    weather_score  = rng_local.uniform(0.0, 10.0, n).round(1)       # 0–10
    pickup_zone    = rng_local.integers(1, 6, n)                    # 1–5
    dropoff_zone   = rng_local.integers(1, 6, n)                    # 1–5

    # --- Derivação do target (fare_amount) ---
    # Tarifa base
    fare = np.full(n, 2.50)

    # Distância (principal driver de preço)
    fare += trip_distance * 1.75

    # Passageiros adicionais
    fare += (num_passengers - 1) * 0.20

    # Horário de pico: manhã (7-9h) e tarde (17-19h)
    rush_mask = ((hour_of_day >= 7) & (hour_of_day <= 9)) | \
                ((hour_of_day >= 17) & (hour_of_day <= 19))
    fare += rush_mask * 2.50

    # Final de semana (sáb=5, dom=6)
    weekend_mask = day_of_week >= 5
    fare += weekend_mask * 1.00

    # Mau tempo aumenta a demanda e o preço (score baixo = mau tempo)
    bad_weather_mask = weather_score < 3.0
    fare += bad_weather_mask * (3.0 - weather_score)

    # Zonas centrais (1 e 2) têm tarifa um pouco maior
    central_zone_mask = (pickup_zone <= 2) | (dropoff_zone <= 2)
    fare += central_zone_mask * 1.50

    # Ruído gaussiano realista (desvio padrão de $1.50)
    fare += rng_local.normal(0, 1.5, n)

    # Garantir valor mínimo de $2.50
    fare = np.maximum(fare, 2.50).round(2)

    df = pd.DataFrame({
        "trip_distance" : trip_distance,
        "num_passengers": num_passengers,
        "hour_of_day"   : hour_of_day,
        "day_of_week"   : day_of_week,
        "weather_score" : weather_score,
        "pickup_zone"   : pickup_zone.astype(int),
        "dropoff_zone"  : dropoff_zone.astype(int),
        "fare_amount"   : fare,
    })
    return df

# Gerar os conjuntos de dados
# Usamos seeds diferentes para garantir que treino e validação/scoring não se sobreponham
df_full    = generate_taxi_data(n_train, seed=random_seed)
df_scoring = generate_taxi_data(n_scoring, seed=random_seed + 999)

# Dividir full em treino (80%) e validação (20%)
split_idx  = int(len(df_full) * 0.8)
df_train   = df_full.iloc[:split_idx].reset_index(drop=True)
df_val     = df_full.iloc[split_idx:].reset_index(drop=True)

print(f"📊 Treino     : {len(df_train):,} registros")
print(f"📊 Validação  : {len(df_val):,} registros")
print(f"📊 Scoring    : {len(df_scoring):,} registros")
print(f"\nEstatísticas descritivas (treino):")
display(df_train.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Persistência no Unity Catalog (Delta Tables)
# MAGIC
# MAGIC Convertemos os DataFrames Pandas para Spark e salvamos como Delta Tables no Unity Catalog.

# COMMAND ----------

# DBTITLE 1, Salvar tabela de treino
train_table = f"{catalog_name}.{schema_name}.taxi_fares_train"

spark.createDataFrame(df_train) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(train_table)

print(f"✅ Tabela de treino salva: {train_table}")
spark.sql(f"SELECT COUNT(*) AS total FROM {train_table}").show()

# COMMAND ----------

# DBTITLE 1, Salvar tabela de validação
val_table = f"{catalog_name}.{schema_name}.taxi_fares_val"

spark.createDataFrame(df_val) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(val_table)

print(f"✅ Tabela de validação salva: {val_table}")
spark.sql(f"SELECT COUNT(*) AS total FROM {val_table}").show()

# COMMAND ----------

# DBTITLE 1, Salvar tabela de scoring (sem a coluna target)
# Na inferência em batch, normalmente não temos o target disponível.
# Removemos 'fare_amount' para simular dados reais de produção.
scoring_table = f"{catalog_name}.{schema_name}.taxi_scoring"

spark.createDataFrame(df_scoring.drop(columns=["fare_amount"])) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(scoring_table)

print(f"✅ Tabela de scoring salva: {scoring_table}")
spark.sql(f"SELECT COUNT(*) AS total FROM {scoring_table}").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificação Final
# MAGIC
# MAGIC Vamos confirmar que as três tabelas foram criadas corretamente no Unity Catalog.

# COMMAND ----------

# DBTITLE 1, Verificar tabelas criadas
print(f"\n{'='*60}")
print(f"TABELAS CRIADAS NO UNITY CATALOG")
print(f"{'='*60}")
tables_created = spark.sql(f"SHOW TABLES IN {catalog_name}.{schema_name}").toPandas()
display(tables_created)

print(f"\n✅ Setup concluído com sucesso!")
print(f"\nPróximo passo: execute o notebook EDA.py para análise exploratória.")
print(f"\nCaminhos das tabelas:")
print(f"  Treino    : {train_table}")
print(f"  Validação : {val_table}")
print(f"  Scoring   : {scoring_table}")

# COMMAND ----------

# DBTITLE 1, Preview das tabelas
print("=== Amostra: Treino ===")
display(spark.table(train_table).limit(5))

print("=== Amostra: Scoring (sem target) ===")
display(spark.table(scoring_table).limit(5))
