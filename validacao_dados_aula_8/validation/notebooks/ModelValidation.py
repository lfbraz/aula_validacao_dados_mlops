# Databricks notebook source
##################################################################################
# Notebook de Validação do Modelo
#
# Executa validação automatizada do modelo recém-treinado antes de promovê-lo
# ao alias "challenger" (e depois "champion" no ModelDeployment).
#
# A validação usa mlflow.evaluate() para computar métricas e verificar
# se o modelo atende aos thresholds definidos em validation/validation.py.
#
# Modos de execução (run_mode):
#   - "disabled" : pula a validação — modelo passa direto para deploy
#   - "dry_run"  : executa validação, mas falhas não bloqueiam o deploy
#   - "enabled"  : executa validação e bloqueia deploy em caso de falha
#
# No contexto de aprendizado, use "dry_run" para visualizar as métricas
# sem risco de bloquear o pipeline.
#
# Parâmetros:
#   * experiment_name                    - Experimento MLflow
#   * run_mode                           - disabled | dry_run | enabled
#   * enable_baseline_comparison         - Comparar com modelo "champion" atual
#   * validation_input                   - Query SQL ou tabela de validação
#   * model_type                         - "regressor" ou "classifier"
#   * targets                            - Coluna alvo no dataset de validação
#   * custom_metrics_loader_function     - Função em validation.py
#   * validation_thresholds_loader_function - Função em validation.py
#   * evaluator_config_loader_function   - Função em validation.py
#   * model_name                         - Nome completo do modelo (UC three-level)
#   * model_version                      - Versão do modelo a validar
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
validation_dir = os.path.dirname(notebook_path)
if validation_dir not in sys.path:
    sys.path.insert(0, validation_dir)

# COMMAND ----------

# DBTITLE 1, Widgets / Parâmetros
dbutils.widgets.text(
    "experiment_name",
    "/dev-validacao_dados_aula_8-experiment",
    "Experimento MLflow",
)
dbutils.widgets.dropdown(
    "run_mode", "dry_run", ["disabled", "dry_run", "enabled"],
    "Modo de execução da validação",
)
dbutils.widgets.dropdown(
    "enable_baseline_comparison", "false", ["true", "false"],
    "Comparar com modelo champion?",
)
# Tabela de validação criada pelo DataSetup.py
dbutils.widgets.text(
    "validation_input",
    "SELECT * FROM dev.validacao_dados_aula_8.taxi_fares_val",
    "Query ou tabela de validação",
)
dbutils.widgets.text("model_type",  "regressor",   "Tipo do modelo")
dbutils.widgets.text("targets",     "fare_amount",  "Coluna alvo")
dbutils.widgets.text("custom_metrics_loader_function",          "custom_metrics",          "Função de métricas customizadas")
dbutils.widgets.text("validation_thresholds_loader_function",   "validation_thresholds",   "Função de thresholds")
dbutils.widgets.text("evaluator_config_loader_function",        "evaluator_config",        "Função de config do evaluator")
dbutils.widgets.text("model_name",    "dev.validacao_dados_aula_8.validacao_dados_aula_8-model", "Nome completo do modelo (UC)")
dbutils.widgets.text("model_version", "", "Versão candidata do modelo")

# COMMAND ----------

# DBTITLE 1, Verificar run_mode
run_mode = dbutils.widgets.get("run_mode").lower()
assert run_mode in ("disabled", "dry_run", "enabled"), \
    f"run_mode inválido: {run_mode}. Use: disabled, dry_run ou enabled."

if run_mode == "disabled":
    print("⏭️  Validação em modo DISABLED — pulando validação e liberando deploy.")
    dbutils.notebook.exit(0)

dry_run = run_mode == "dry_run"
if dry_run:
    print("🔍 Validação em modo DRY_RUN — falhas serão logadas, mas NÃO bloquearão o deploy.")
else:
    print("🛡️  Validação em modo ENABLED — falhas bloquearão o deploy.")

# COMMAND ----------

# DBTITLE 1, Importar dependências
import importlib
import mlflow
import os
import tempfile
import traceback

from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
mlflow.set_registry_uri("databricks-uc")

# Configurar experimento MLflow
experiment_name = dbutils.widgets.get("experiment_name")
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1, Recuperar informações do modelo (via task values ou widgets)
# Durante execução no workflow, o modelo é passado como task value da task "Train"
# Durante execução manual, usamos os widgets
model_uri     = dbutils.jobs.taskValues.get("Train", "model_uri",     debugValue="")
model_name    = dbutils.jobs.taskValues.get("Train", "model_name",    debugValue="")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="")

if model_uri == "":
    model_name    = dbutils.widgets.get("model_name")
    model_version = dbutils.widgets.get("model_version")

    if not model_version:
        versions = client.search_model_versions(f"name='{model_name}'")
        assert versions, f"Nenhuma versão encontrada para o modelo '{model_name}'"
        model_version = str(max(int(mv.version) for mv in versions))
        print(f"ℹ️  model_version não informado — usando última versão: {model_version}")

    model_uri = f"models:/{model_name}/{model_version}"

baseline_model_uri = f"models:/{model_name}@champion"

assert model_uri     != "", "model_uri não especificado"
assert model_name    != "", "model_name não especificado"
assert model_version != "", "model_version não especificado"

print(f"Modelo candidato : {model_uri}")
print(f"Modelo baseline  : {baseline_model_uri}")

# COMMAND ----------

# DBTITLE 1, Carregar dados de validação e configurações
enable_baseline_comparison = dbutils.widgets.get("enable_baseline_comparison") == "true"
validation_input           = dbutils.widgets.get("validation_input")
model_type                 = dbutils.widgets.get("model_type")
targets                    = dbutils.widgets.get("targets")

# Carregar dados de validação
data = spark.sql(validation_input)
print(f"📊 Dados de validação carregados: {data.count():,} registros")
display(data.limit(5))

# COMMAND ----------

# DBTITLE 1, Carregar funções de validação do módulo validation.py
custom_metrics_fn    = dbutils.widgets.get("custom_metrics_loader_function")
thresholds_fn        = dbutils.widgets.get("validation_thresholds_loader_function")
evaluator_config_fn  = dbutils.widgets.get("evaluator_config_loader_function")

validation_module = importlib.import_module("validation")

custom_metrics_loader       = getattr(validation_module, custom_metrics_fn)
validation_thresholds_loader = getattr(validation_module, thresholds_fn)
evaluator_config_loader     = getattr(validation_module, evaluator_config_fn)

custom_metrics       = custom_metrics_loader()
validation_thresholds = validation_thresholds_loader()
evaluator_config     = evaluator_config_loader()

print("✅ Funções de validação carregadas:")
print(f"   Métricas customizadas : {[m.name for m in custom_metrics]}")
print(f"   Thresholds            : {list(validation_thresholds.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Executar Avaliação com mlflow.evaluate()
# MAGIC
# MAGIC O `mlflow.evaluate()` é a função central da validação. Ele:
# MAGIC 1. Carrega o modelo a partir do `model_uri`
# MAGIC 2. Faz previsões sobre os dados de validação
# MAGIC 3. Computa métricas built-in (RMSE, MAE, R², etc.)
# MAGIC 4. Computa métricas customizadas definidas em `validation.py`
# MAGIC 5. Verifica se todas as métricas atendem aos thresholds
# MAGIC 6. Se `enable_baseline_comparison=True`, compara com o modelo champion

# COMMAND ----------

# DBTITLE 1, Funções auxiliares
def get_run_link(run_info):
    return "[Run](#mlflow/experiments/{0}/runs/{1})".format(
        run_info.experiment_id, run_info.run_id
    )

def get_training_run(model_name, model_version):
    version = client.get_model_version(model_name, model_version)
    if not version.run_id:
        return None
    return mlflow.get_run(run_id=version.run_id)

def generate_run_name(training_run):
    return None if not training_run else training_run.info.run_name + "-validation"

def generate_description(training_run):
    return (
        None if not training_run
        else f"Detalhes do treino: {get_run_link(training_run.info)}\n"
    )

def log_to_model_description(run, success):
    run_link    = get_run_link(run.info)
    description = client.get_model_version(model_name, model_version).description or ""
    status      = "✅ PASSOU" if success else "❌ FALHOU"
    if description:
        description += "\n\n---\n\n"
    description += f"Status da Validação: {status}\nDetalhes: {run_link}"
    client.update_model_version(name=model_name, version=model_version, description=description)

# COMMAND ----------

# DBTITLE 1, Executar avaliação do modelo
training_run = get_training_run(model_name, model_version)
evaluators   = "default"

with mlflow.start_run(
    run_name=generate_run_name(training_run),
    description=generate_description(training_run),
) as run, tempfile.TemporaryDirectory() as tmp_dir:

    # Log dos thresholds como artefato
    thresholds_file = os.path.join(tmp_dir, "validation_thresholds.txt")
    with open(thresholds_file, "w") as f:
        f.write("THRESHOLDS DE VALIDAÇÃO\n")
        f.write("=" * 40 + "\n")
        for metric, threshold in validation_thresholds.items():
            f.write(f"  {metric:30s}: {threshold}\n")
    mlflow.log_artifact(thresholds_file)

    try:
        print("🚀 Iniciando mlflow.evaluate()...")
        eval_result = mlflow.evaluate(
            model=model_uri,
            data=data,
            targets=targets,
            model_type=model_type,
            evaluators=evaluators,
            extra_metrics=custom_metrics,
            evaluator_config=evaluator_config,
        )

        # Verificar thresholds manualmente
        violations = []
        for metric_name, t in validation_thresholds.items():
            actual = eval_result.metrics.get(metric_name)
            if actual is None:
                continue
            passed = actual >= t.threshold if t.greater_is_better else actual <= t.threshold
            op = ">=" if t.greater_is_better else "<="
            status = "✅" if passed else "❌"
            print(f"   {status} {metric_name:<35} = {actual:.4f}  ({op} {t.threshold})")
            if not passed:
                violations.append(f"{metric_name}: {actual:.4f} (esperado {op} {t.threshold})")

        if violations:
            raise Exception("Thresholds violados:\n" + "\n".join(f"  - {v}" for v in violations))

        # Log das métricas como artefato para fácil visualização
        baseline_metrics = getattr(eval_result, "baseline_model_metrics", {}) or {}
        metrics_file = os.path.join(tmp_dir, "metrics_comparison.txt")
        with open(metrics_file, "w") as f:
            f.write(f"{'Métrica':<30}  {'Candidato':<20}  {'Baseline'}\n")
            f.write("-" * 70 + "\n")
            for metric_name, candidate_val in eval_result.metrics.items():
                baseline_val = baseline_metrics.get(metric_name, "N/A")
                if isinstance(baseline_val, float):
                    mlflow.log_metric(f"baseline_{metric_name}", baseline_val)
                f.write(f"{metric_name:<30}  {str(candidate_val):<20}  {str(baseline_val)}\n")
        mlflow.log_artifact(metrics_file)

        log_to_model_description(run, True)

        # Atribuir alias "challenger" — indica que passou na validação
        print("\n✅ Todas as validações passaram!")
        print(f"   Atribuindo alias 'challenger' à versão {model_version}...")
        client.set_registered_model_alias(model_name, "challenger", model_version)
        print(f"   Alias 'challenger' atribuído com sucesso.")

    except Exception as err:
        log_to_model_description(run, False)

        # Salvar detalhes do erro
        error_file = os.path.join(tmp_dir, "validation_error.txt")
        with open(error_file, "w") as f:
            f.write(f"VALIDAÇÃO FALHOU\n{'='*40}\n")
            f.write(str(err) + "\n\n")
            f.write(traceback.format_exc())
        mlflow.log_artifact(error_file)

        if not dry_run:
            print(f"\n❌ Validação falhou em modo ENABLED — bloqueando deploy.")
            raise err
        else:
            print(f"\n⚠️  Validação falhou em modo DRY_RUN — deploy não será bloqueado.")
            print(f"   Erro: {err}")
