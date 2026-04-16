# Databricks notebook source
##################################################################################
# Helper notebook to transition the model stage. This notebook is run
# after the Train.py notebook as part of a multi-task job, in order to transition model
# to target stage after training completes.
#
# This notebook has the following parameters:
#
#  * env (required)  - String name of the current environment for model deployment.
#  * model_uri       - URI of the model to deploy (read as task value from Train task).
##################################################################################

# COMMAND ----------

import os
spark.range(1).collect()
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# MAGIC %pip install -r $notebook_path/../../../requirements.txt

# COMMAND ----------

import os, sys
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
deploy_dir = os.path.dirname(notebook_path)  # deployment/model_deployment/
if deploy_dir not in sys.path:
    sys.path.insert(0, deploy_dir)

# COMMAND ----------

dbutils.widgets.dropdown("env", "None", ["None", "dev", "staging", "prod"], "Environment Name")
dbutils.widgets.text("model_name", "dev.validacao_dados_aula_8.validacao_dados_aula_8-model", "Nome completo do modelo (UC)")

# COMMAND ----------

from deploy import deploy
from mlflow.tracking import MlflowClient

model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
env = dbutils.widgets.get("env")
assert env != "None", "env notebook parameter must be specified"

if not model_uri:
    model_name = dbutils.widgets.get("model_name")
    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(f"name='{model_name}'")
    assert versions, f"Nenhuma versão encontrada para o modelo '{model_name}'"
    latest_version = str(max(int(mv.version) for mv in versions))
    model_uri = f"models:/{model_name}/{latest_version}"
    print(f"ℹ️  model_uri não recebido via task values — usando última versão: {model_uri}")

deploy(model_uri, env)

# COMMAND ----------

print(f"Successfully completed model deployment for {model_uri}")
