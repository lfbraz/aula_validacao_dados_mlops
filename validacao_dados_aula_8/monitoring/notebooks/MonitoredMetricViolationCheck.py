# Databricks notebook source
##################################################################################
# This notebook runs a sql query and set the result as job task value
#
# This notebook has the following parameters:
#
#  * table_name_under_monitor (required)  - The name of a table that is currently being monitored
#  * metric_to_monitor (required)  - Metric to be monitored for threshold violation
#  * metric_violation_threshold (required)  - Threshold value for metric violation
#  * num_evaluation_windows (required)  - Number of windows to check for violation
#  * num_violation_windows (required)  - Number of windows that need to violate the threshold
##################################################################################

# COMMAND ----------

import os
spark.range(1).collect()
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# COMMAND ----------

# MAGIC %pip install -r $notebook_path/../../requirements.txt

# COMMAND ----------

import os, sys
notebook_path = '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
monitoring_dir = os.path.dirname(notebook_path)  # monitoring/
if monitoring_dir not in sys.path:
    sys.path.insert(0, monitoring_dir)

# COMMAND ----------

dbutils.widgets.text(
    "table_name_under_monitor", "dev.validacao_dados_aula_8.predictions", label="Full (three-Level) table name"
)
dbutils.widgets.text(
    "metric_to_monitor", "root_mean_squared_error", label="Metric to be monitored for threshold violation"
)
dbutils.widgets.text(
    "metric_violation_threshold", "100", label="Threshold value for metric violation"
)
dbutils.widgets.text(
    "num_evaluation_windows", "5", label="Number of windows to check for violation"
)
dbutils.widgets.text(
    "num_violation_windows", "2", label="Number of windows that need to violate the threshold"
)

# COMMAND ----------

from metric_violation_check_query import sql_query

table_name_under_monitor   = dbutils.widgets.get("table_name_under_monitor")
metric_to_monitor          = dbutils.widgets.get("metric_to_monitor")
metric_violation_threshold = dbutils.widgets.get("metric_violation_threshold")
num_evaluation_windows     = dbutils.widgets.get("num_evaluation_windows")
num_violation_windows      = dbutils.widgets.get("num_violation_windows")

profile_metrics_table = f"{table_name_under_monitor}_profile_metrics"

if not spark.catalog.tableExists(profile_metrics_table):
    print(f"⚠️  Tabela '{profile_metrics_table}' ainda não existe — o monitor ainda não executou.")
    print("   Definindo is_metric_violated=False e encerrando sem erro.")
    is_metric_violated = False
else:
    formatted_sql_query = sql_query.format(
        table_name_under_monitor=table_name_under_monitor,
        metric_to_monitor=metric_to_monitor,
        metric_violation_threshold=metric_violation_threshold,
        num_evaluation_windows=num_evaluation_windows,
        num_violation_windows=num_violation_windows,
    )
    is_metric_violated = bool(spark.sql(formatted_sql_query).toPandas()["query_result"][0])

print(f"is_metric_violated: {is_metric_violated}")
dbutils.jobs.taskValues.set("is_metric_violated", is_metric_violated)
