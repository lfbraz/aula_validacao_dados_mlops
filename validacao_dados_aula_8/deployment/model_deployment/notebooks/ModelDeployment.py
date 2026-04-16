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

# COMMAND ----------

from deploy import deploy

model_uri = dbutils.jobs.taskValues.get("Train", "model_uri", debugValue="")
env = dbutils.widgets.get("env")
assert env != "None", "env notebook parameter must be specified"
assert model_uri != "", "model_uri notebook parameter must be specified"
deploy(model_uri, env)

# COMMAND ----------

print(f"Successfully completed model deployment for {model_uri}")
