# Databricks notebook source
##################################################################################
# Notebook de Inferência em Batch
#
# Aplica o modelo "champion" (produção) para gerar previsões em lote sobre
# uma tabela Delta de entrada, salvando os resultados em outra tabela Delta.
#
# Fluxo:
#   1. Carrega o modelo champion do Unity Catalog Model Registry
#   2. Lê os dados de entrada da tabela input_table_name
#   3. Aplica o modelo como Spark UDF (escalonável para grandes volumes)
#   4. Salva as previsões com metadata (versão do modelo, timestamp)
#
# Este notebook é executado pelo job "batch_inference_job" configurado em
# resources/batch-inference-workflow-resource.yml
#
# Parâmetros:
#   * env               - Ambiente (dev, staging, prod)
#   * input_table_name  - Tabela Delta com os dados de scoring (sem target)
#   * output_table_name - Tabela Delta de destino das previsões
#   * model_name        - Nome completo do modelo (three-level UC name)
##################################################################################

# COMMAND ----------

dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
dbutils.widgets.text("input_table_name",  "", label="Tabela de entrada (scoring)")
dbutils.widgets.text("output_table_name", "", label="Tabela de saída (previsões)")
dbutils.widgets.text(
    "model_name",
    "dev.validacao_dados_aula_8.validacao_dados_aula_8-model",
    label="Nome completo do modelo (three-level UC name)",
)

# COMMAND ----------

import os
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1, Definir variáveis
env               = dbutils.widgets.get("env")
input_table_name  = dbutils.widgets.get("input_table_name")
output_table_name = dbutils.widgets.get("output_table_name")
model_name        = dbutils.widgets.get("model_name")

assert input_table_name  != "", "input_table_name não especificado"
assert output_table_name != "", "output_table_name não especificado"
assert model_name        != "", "model_name não especificado"

# O alias "champion" aponta para o modelo que passou por validação e foi promovido
alias     = "champion"
model_uri = f"models:/{model_name}@{alias}"

print(f"Ambiente         : {env}")
print(f"Tabela entrada   : {input_table_name}")
print(f"Tabela saída     : {output_table_name}")
print(f"Modelo URI       : {model_uri}")

# COMMAND ----------

# DBTITLE 1, Obter versão do modelo champion
from mlflow import MlflowClient

client        = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

print(f"✅ Modelo champion: versão {model_version}")

# COMMAND ----------

# DBTITLE 1, Timestamp da execução
from datetime import datetime
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Timestamp: {ts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inferência em Batch com Spark UDF
# MAGIC
# MAGIC Utilizamos `mlflow.pyfunc.spark_udf()` para carregar o modelo como uma UDF Spark.
# MAGIC Isso permite aplicar o modelo em paralelo sobre toda a tabela, aproveitando
# MAGIC o poder de processamento distribuído do Databricks.

# COMMAND ----------

# DBTITLE 1, Executar inferência em batch
from predict import predict_batch

print(f"🚀 Iniciando inferência em batch...")
predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
print(f"✅ Previsões salvas em: {output_table_name}")

dbutils.notebook.exit(output_table_name)
