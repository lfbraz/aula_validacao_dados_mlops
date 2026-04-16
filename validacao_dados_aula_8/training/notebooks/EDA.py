# Databricks notebook source
##################################################################################
# Notebook de EDA — Análise Exploratória de Dados
#
# Antes de treinar qualquer modelo, é fundamental entender os dados.
# Este notebook demonstra uma EDA básica sobre o dataset sintético de
# corridas de táxi criado no DataSetup.py.
#
# Etapas cobertas:
#   1. Carregamento e inspeção inicial dos dados
#   2. Estatísticas descritivas e detecção de nulos/duplicatas
#   3. Distribuição do target (fare_amount)
#   4. Análise de correlações
#   5. Análise por features categóricas
#   6. Conclusões para o processo de treinamento
#
# Parâmetros:
#   * catalog_name        - Catalog Unity Catalog (ex: dev)
#   * schema_name         - Schema (padrão: validacao_dados_aula_8)
#   * training_data_path  - Caminho completo da tabela de treino (three-level)
##################################################################################

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %pip install seaborn --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1, Parâmetros
dbutils.widgets.text("catalog_name", "dev", "Catalog Name")
dbutils.widgets.text("schema_name", "validacao_dados_aula_8", "Schema Name")
dbutils.widgets.text(
    "training_data_path",
    "dev.validacao_dados_aula_8.taxi_fares_train",
    "Caminho da tabela de treino",
)

catalog_name       = dbutils.widgets.get("catalog_name")
schema_name        = dbutils.widgets.get("schema_name")
training_data_path = dbutils.widgets.get("training_data_path")

print(f"Tabela: {training_data_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Carregamento e Inspeção Inicial
# MAGIC
# MAGIC Carregamos os dados diretamente do Delta Lake (Unity Catalog) e fazemos uma inspeção inicial.

# COMMAND ----------

# DBTITLE 1, Carregar dados
df_spark = spark.table(training_data_path)

print(f"✅ Tabela carregada: {training_data_path}")
print(f"   Linhas  : {df_spark.count():,}")
print(f"   Colunas : {len(df_spark.columns)}")
print(f"\nSchema:")
df_spark.printSchema()

# COMMAND ----------

# DBTITLE 1, Preview dos dados
display(df_spark.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Qualidade dos Dados
# MAGIC
# MAGIC Verificamos nulos, duplicatas e consistência dos dados antes de treinar o modelo.

# COMMAND ----------

# DBTITLE 1, Verificar valores nulos
import pyspark.sql.functions as F

print("=== Valores Nulos por Coluna ===")
null_counts = df_spark.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_spark.columns
])
display(null_counts)

# COMMAND ----------

# DBTITLE 1, Verificar duplicatas
total       = df_spark.count()
unique_rows = df_spark.dropDuplicates().count()
duplicates  = total - unique_rows

print(f"Total de registros  : {total:,}")
print(f"Registros únicos    : {unique_rows:,}")
print(f"Duplicatas          : {duplicates:,}")

if duplicates == 0:
    print("\n✅ Nenhuma duplicata encontrada!")
else:
    print(f"\n⚠️  {duplicates} duplicatas encontradas — considerar remoção antes do treino.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estatísticas Descritivas

# COMMAND ----------

# DBTITLE 1, Estatísticas descritivas
display(df_spark.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Análise do Target: `fare_amount`
# MAGIC
# MAGIC O target é o valor em USD da corrida de táxi. Entender sua distribuição é fundamental.

# COMMAND ----------

# DBTITLE 1, Distribuição do target
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = df_spark.toPandas()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histograma
axes[0].hist(df["fare_amount"], bins=50, color="#1565C0", alpha=0.8, edgecolor="white")
axes[0].set_title("Distribuição de fare_amount", fontsize=14)
axes[0].set_xlabel("Valor da corrida (USD)")
axes[0].set_ylabel("Frequência")
axes[0].axvline(df["fare_amount"].median(), color="red", linestyle="--", label=f"Mediana: ${df['fare_amount'].median():.2f}")
axes[0].legend()

# Boxplot
axes[1].boxplot(df["fare_amount"], vert=True, patch_artist=True,
                boxprops=dict(facecolor="#1565C0", alpha=0.7))
axes[1].set_title("Boxplot de fare_amount", fontsize=14)
axes[1].set_ylabel("Valor da corrida (USD)")

plt.tight_layout()
plt.show()

print(f"\nEstatísticas do target:")
print(df["fare_amount"].describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Análise de Correlações
# MAGIC
# MAGIC Observamos como cada feature se correlaciona com o target.
# MAGIC Esperamos que `trip_distance` tenha alta correlação (por design do dataset sintético).

# COMMAND ----------

# DBTITLE 1, Matriz de correlação
fig, ax = plt.subplots(figsize=(10, 8))

corr_matrix = df.corr(numeric_only=True)

# Usar seaborn para um heatmap mais visual
mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    ax=ax,
    linewidths=0.5,
    square=True
)
ax.set_title("Matriz de Correlação — Features × Target", fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1, Correlação com o target (ordenada)
target_corr = corr_matrix["fare_amount"].drop("fare_amount").sort_values(ascending=False)
print("Correlação com fare_amount (ordenada):\n")
print(target_corr.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Análise por Features Categóricas/Discretas

# COMMAND ----------

# DBTITLE 1, Fare amount por hora do dia
plt.figure(figsize=(14, 5))
hourly_fare = df.groupby("hour_of_day")["fare_amount"].mean()
plt.bar(hourly_fare.index, hourly_fare.values, color="#1565C0", alpha=0.8, edgecolor="white")
plt.title("Valor Médio da Corrida por Hora do Dia", fontsize=14)
plt.xlabel("Hora do Dia")
plt.ylabel("Fare Amount Médio (USD)")
plt.xticks(range(0, 24))
plt.axvline(x=7.5, color="red", linestyle="--", alpha=0.5, label="Horário de pico (manhã)")
plt.axvline(x=18, color="orange", linestyle="--", alpha=0.5, label="Horário de pico (tarde)")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1, Fare amount por dia da semana
day_names = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
plt.figure(figsize=(10, 5))
weekly_fare = df.groupby("day_of_week")["fare_amount"].mean()
bars = plt.bar(range(7), weekly_fare.values, color=["#1565C0"]*5 + ["#E53935"]*2, alpha=0.8, edgecolor="white")
plt.title("Valor Médio da Corrida por Dia da Semana", fontsize=14)
plt.xlabel("Dia da Semana")
plt.ylabel("Fare Amount Médio (USD)")
plt.xticks(range(7), day_names)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1, Relação entre distância e valor da corrida (scatterplot)
sample = df.sample(min(1000, len(df)), random_state=42)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    sample["trip_distance"],
    sample["fare_amount"],
    c=sample["hour_of_day"],
    cmap="RdYlBu_r",
    alpha=0.6,
    s=20
)
plt.colorbar(scatter, label="Hora do dia")
plt.title("Distância × Valor da Corrida (colorido por hora)", fontsize=14)
plt.xlabel("Distância da Corrida (km)")
plt.ylabel("Fare Amount (USD)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1, Fare amount por pickup zone
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pickup zone
zone_fare_pickup  = df.groupby("pickup_zone")["fare_amount"].mean()
axes[0].bar(zone_fare_pickup.index, zone_fare_pickup.values, color="#1565C0", alpha=0.8, edgecolor="white")
axes[0].set_title("Fare Médio por Zona de Origem", fontsize=13)
axes[0].set_xlabel("Pickup Zone")
axes[0].set_ylabel("Fare Amount Médio (USD)")

# Dropoff zone
zone_fare_dropoff = df.groupby("dropoff_zone")["fare_amount"].mean()
axes[1].bar(zone_fare_dropoff.index, zone_fare_dropoff.values, color="#E53935", alpha=0.8, edgecolor="white")
axes[1].set_title("Fare Médio por Zona de Destino", fontsize=13)
axes[1].set_xlabel("Dropoff Zone")
axes[1].set_ylabel("Fare Amount Médio (USD)")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1, Impacto do clima no valor da corrida
# Agrupamos weather_score em categorias para visualização
df["weather_category"] = pd.cut(
    df["weather_score"],
    bins=[-0.1, 2.9, 5.9, 10.1],
    labels=["Mau (0-3)", "Moderado (3-6)", "Bom (6-10)"]
)
weather_fare = df.groupby("weather_category")["fare_amount"].mean()

plt.figure(figsize=(8, 5))
plt.bar(weather_fare.index, weather_fare.values,
        color=["#E53935", "#FFA726", "#66BB6A"], alpha=0.85, edgecolor="white")
plt.title("Impacto do Clima no Valor Médio da Corrida", fontsize=14)
plt.xlabel("Categoria Climática")
plt.ylabel("Fare Amount Médio (USD)")
plt.tight_layout()
plt.show()

# COMMAND ----------

print("✅ EDA concluída! Acesse o notebook Train.py para iniciar o treinamento do modelo.")
