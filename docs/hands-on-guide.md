# Guia Prático de MLOps com Databricks MLOps Stacks

> **Aula 8 — Validação de Dados e MLOps em Produção**
> Este guia acompanha a demonstração prática da aula. Os scripts referenciados
> estão no diretório `validacao_dados_aula_8/` deste repositório.

---

## Índice

1. [HANDS ON — Pipeline MLOps com MLOps Stacks](#hands-on)
   - [Visão Geral da Arquitetura](#visão-geral)
   - [Passo 0: Configuração do Projeto](#passo-0-configuração-do-projeto)
   - [Passo 1: Setup dos Dados (DataSetup)](#passo-1-setup-dos-dados)
   - [Passo 2: Análise Exploratória (EDA)](#passo-2-análise-exploratória)
   - [Passo 3: Treinamento do Modelo](#passo-3-treinamento-do-modelo)
   - [Passo 4: Validação do Modelo](#passo-4-validação-do-modelo)
   - [Passo 5: Deploy do Modelo](#passo-5-deploy-do-modelo)
   - [Passo 6: Inferência em Batch](#passo-6-inferência-em-batch)
   - [Passo 7: Monitoramento](#passo-7-monitoramento)
2. [SAIBA MAIS — Databricks Asset Bundles](#saiba-mais)

---

## HANDS ON — Pipeline MLOps com MLOps Stacks {#hands-on}

### Visão Geral da Arquitetura {#visão-geral}

O **Databricks MLOps Stacks** é um template opinativo que implementa um pipeline
de Machine Learning production-ready com as seguintes etapas:

```
╔══════════════════════════════════════════════════════════════════════╗
║               PIPELINE DE TREINAMENTO + VALIDAÇÃO + DEPLOY          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   [DataSetup] → [EDA] → [Train] → [Validate] → [Deploy]             ║
║        ↓                   ↓           ↓           ↓                ║
║   Delta Tables         MLflow Exp.  Challenger   Champion            ║
║   (Unity Catalog)      (Metrics)    (alias)      (alias)            ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                     PIPELINE DE INFERÊNCIA EM BATCH                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   [taxi_scoring] → [BatchInference (@champion)] → [predictions]     ║
║   (Delta Table)       (Spark UDF + MLflow)        (Delta Table)     ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         PIPELINE DE MONITORAMENTO                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   [predictions] → [Lakehouse Monitor] → [MetricCheck] → [Retrain?]  ║
║   (Delta Table)    (perfil estatístico)   (threshold)   (trigger)   ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Cenário da demonstração:** previsão do valor de corridas de táxi (`fare_amount`)
com um dataset sintético criado para fins didáticos. O modelo é um regressor
LightGBM treinado sobre 7 features numéricas.

**Estrutura de diretórios do projeto:**

```
validacao_dados_aula_8/
├── setup/notebooks/DataSetup.py          ← NOVO: cria tabelas com dados sintéticos
├── training/notebooks/
│   ├── EDA.py                            ← NOVO: análise exploratória
│   └── Train.py                          ← Treinamento LightGBM + MLflow
├── validation/
│   ├── validation.py                     ← Métricas customizadas e thresholds
│   └── notebooks/ModelValidation.py      ← Executa mlflow.evaluate()
├── deployment/
│   ├── batch_inference/notebooks/BatchInference.py
│   └── model_deployment/notebooks/ModelDeployment.py
├── monitoring/notebooks/MonitoredMetricViolationCheck.py
├── resources/                            ← Databricks Asset Bundles YAMLs
│   ├── model-workflow-resource.yml
│   ├── batch-inference-workflow-resource.yml
│   └── monitoring-resource.yml
└── databricks.yml                        ← Bundle raiz
```

---

### Passo 0: Configuração do Projeto {#passo-0-configuração-do-projeto}

#### Pré-requisitos

Antes de executar os notebooks, certifique-se de que:

1. **Databricks CLI instalado e configurado:**
   ```bash
   pip install databricks-cli
   databricks configure --token
   ```

2. **Catalog criado no Unity Catalog:** o projeto usa o catalog `dev` por padrão.
   O schema `validacao_dados_aula_8` será criado automaticamente pelo DataSetup.

3. **Permissões no Unity Catalog:**
   - `CREATE TABLE` e `MODIFY` no schema `dev.validacao_dados_aula_8`
   - `CREATE MODEL` para registrar no Model Registry

4. **Variáveis do bundle configuradas** em `databricks.yml`:
   ```yaml
   targets:
     dev:
       variables:
         catalog_name: dev    # ← altere conforme seu catalog
       workspace:
         host: https://seu-workspace.azuredatabricks.net
   ```

#### Deploy com Databricks Asset Bundles

O MLOps Stacks usa Databricks Asset Bundles (DABs) para gerenciar todos os
recursos de forma declarativa. Com um único comando, você cria os jobs,
experimentos MLflow e configurações de monitoramento:

```bash
# No diretório validacao_dados_aula_8/
cd validacao_dados_aula_8

# Validar a configuração do bundle
databricks bundle validate

# Fazer deploy para o ambiente dev
databricks bundle deploy --target dev

# Listar os jobs criados
databricks bundle run --help
```

---

### Passo 1: Setup dos Dados {#passo-1-setup-dos-dados}

**Notebook:** `setup/notebooks/DataSetup.py`

Este notebook cria um dataset sintético de corridas de táxi no Unity Catalog,
simulando um cenário real de produção com dados já disponíveis no Delta Lake.

**O que ele faz:**

1. Cria o schema `dev.validacao_dados_aula_8` (se não existir)
2. Gera 5.000 registros de treino com 7 features
3. Separa 20% para validação
4. Gera 500 registros de scoring (sem o target)
5. Salva tudo como Delta Tables no Unity Catalog

**Execute no Databricks:**

Abra o notebook no Workspace e execute com os parâmetros padrão, ou configure
via widgets:

| Widget | Padrão | Descrição |
|--------|--------|-----------|
| `catalog_name` | `dev` | Catalog Unity Catalog |
| `schema_name` | `validacao_dados_aula_8` | Schema para as tabelas |
| `n_train_samples` | `5000` | Registros de treino |
| `n_scoring_samples` | `500` | Registros para inferência |

**Tabelas criadas:**

```
dev.validacao_dados_aula_8.taxi_fares_train   (4.000 registros)
dev.validacao_dados_aula_8.taxi_fares_val     (1.000 registros)
dev.validacao_dados_aula_8.taxi_scoring       (500 registros, sem fare_amount)
```

**Schema das tabelas de treino/validação:**

```
trip_distance    DOUBLE   -- distância em km (0.5–30)
num_passengers   INT      -- passageiros (1–6)
hour_of_day      INT      -- hora de início (0–23)
day_of_week      INT      -- dia da semana (0=seg, 6=dom)
weather_score    DOUBLE   -- qualidade climática (0–10)
pickup_zone      INT      -- zona de origem (1–5)
dropoff_zone     INT      -- zona de destino (1–5)
fare_amount      DOUBLE   -- TARGET: valor da corrida em USD
```

**Como foi gerado o target (fare_amount):**

```python
fare = 2.50                              # tarifa base
     + trip_distance × 1.75             # distância é o principal driver
     + (num_passengers - 1) × 0.20      # passageiros extras
     + 2.50 se horário de pico          # 7–9h ou 17–19h
     + 1.00 se final de semana          # sáb/dom
     + até 3.00 se mau tempo            # weather_score < 3
     + 1.50 se zona central             # zones 1 ou 2
     + ruído gaussiano (σ = 1.5)
```

---

### Passo 2: Análise Exploratória {#passo-2-análise-exploratória}

**Notebook:** `training/notebooks/EDA.py`

A EDA (Exploratory Data Analysis) é a primeira etapa do ciclo de ML. Entender
os dados antes de treinar o modelo é fundamental para tomar boas decisões.

**O que o notebook faz:**

1. **Verificação de qualidade:** nulos, duplicatas, tipos de dados
2. **Estatísticas descritivas:** distribuição do target e das features
3. **Análise de correlações:** matriz de correlação com heatmap
4. **Análise por hora:** tarifa média por hora do dia (efeito de pico)
5. **Análise por dia da semana:** diferença entre dias úteis e fim de semana
6. **Análise por zona:** impacto das zonas de origem/destino
7. **Impacto do clima:** como `weather_score` afeta o preço

**Principais achados esperados:**

- `trip_distance` tem correlação ~0.85 com `fare_amount` (por design)
- Horários de pico (7–9h, 17–19h) elevam o preço em ~$2.50
- Mau tempo (`weather_score < 3`) eleva o preço em até $3.00
- Zonas centrais (1 e 2) têm tarifa ligeiramente maior

**Execute com o comando:**
```bash
databricks bundle run validacao_dados_aula_8-eda-job --target dev
```
Ou abra diretamente no Workspace Databricks.

---

### Passo 3: Treinamento do Modelo {#passo-3-treinamento-do-modelo}

**Notebook:** `training/notebooks/Train.py`

Este é o coração do pipeline. Treina um modelo LightGBM de regressão e registra
tudo no MLflow Tracking e no Unity Catalog Model Registry.

**O que acontece nesta etapa:**

```
Dados Delta (Unity Catalog)
        ↓
Feature Engineering
  is_rush_hour, is_weekend, is_bad_weather, distance_x_passengers
        ↓
Train/Test Split (80/20)
        ↓
LightGBM Training com mlflow.lightgbm.autolog()
  - Hiperparâmetros logados automaticamente
  - Métricas (RMSE, MAE) logadas por iteração
  - Modelo salvo como artefato MLflow
        ↓
Registro no Unity Catalog Model Registry
  dev.validacao_dados_aula_8.validacao_dados_aula_8-model (versão N)
        ↓
Task Values → Model Validation
  model_uri, model_name, model_version
```

**Features usadas pelo modelo:**

| Feature | Tipo | Origem |
|---------|------|--------|
| `trip_distance` | original | dataset |
| `num_passengers` | original | dataset |
| `hour_of_day` | original | dataset |
| `day_of_week` | original | dataset |
| `weather_score` | original | dataset |
| `pickup_zone` | original | dataset |
| `dropoff_zone` | original | dataset |
| `is_rush_hour` | engineered | `hour_of_day in [7-9, 17-19]` |
| `is_weekend` | engineered | `day_of_week >= 5` |
| `is_bad_weather` | engineered | `weather_score < 3` |
| `distance_x_passengers` | engineered | `trip_distance × num_passengers` |

**Hiperparâmetros padrão (didáticos):**

```python
params = {
    "objective"    : "regression",
    "metric"       : ["rmse", "mae"],
    "num_leaves"   : 31,
    "learning_rate": 0.05,
    "n_estimators" : 200,
}
```

**Executar o job completo (Train → Validate → Deploy):**

```bash
databricks bundle run model_training_job --target dev
```

**Executar apenas o notebook de treino:**

Abra `training/notebooks/Train.py` no Databricks Workspace e clique em "Run All".

**Visualizar resultados no MLflow:**

Após o treino, acesse o experimento `/dev-validacao_dados_aula_8-experiment`
no MLflow UI para ver:
- Hiperparâmetros
- Curvas de aprendizado (loss por iteração)
- Métricas de avaliação (RMSE, MAE, R²)
- Feature importance

---

### Passo 4: Validação do Modelo {#passo-4-validação-do-modelo}

**Notebooks/Scripts:** `validation/notebooks/ModelValidation.py` e `validation/validation.py`

Antes de enviar o modelo para produção, validamos automaticamente se ele atende
aos critérios mínimos de qualidade definidos em `validation.py`.

**Como funciona:**

```
Modelo candidato (version N)   +   Tabela de validação
         ↓                              ↓
              mlflow.evaluate()
                     ↓
         Métricas built-in (RMSE, MAE, R²)
         + Métricas customizadas (MAPE, large_error_rate)
                     ↓
         Verificação dos Thresholds
                /           \
          PASSOU             FALHOU
            ↓                   ↓
      alias "challenger"   raise Exception (se enabled)
      → próxima etapa      ou log + continua (se dry_run)
```

**Thresholds configurados (`validation/validation.py`):**

```python
def validation_thresholds():
    return {
        "root_mean_squared_error": MetricThreshold(threshold=4.00, greater_is_better=False),
        "mean_absolute_error"    : MetricThreshold(threshold=3.00, greater_is_better=False),
        "r2_score"               : MetricThreshold(threshold=0.85, greater_is_better=True),
        "mape"                   : MetricThreshold(threshold=0.25, greater_is_better=False),
        "large_error_rate"       : MetricThreshold(threshold=0.20, greater_is_better=False),
    }
```

**Métricas customizadas definidas:**

```python
def custom_metrics():
    # MAPE — erro relativo percentual
    def mean_absolute_percentage_error(eval_df, _builtin_metrics):
        ...
    # large_error_rate — % de erros > $5
    def large_error_rate(eval_df, _builtin_metrics):
        ...
```

**Modos de execução:**

| Modo | Comportamento | Quando usar |
|------|---------------|-------------|
| `disabled` | Pula a validação | Testes rápidos |
| `dry_run` | Valida mas não bloqueia | **Demonstração em aula** |
| `enabled` | Bloqueia deploy se falhar | **Produção** |

**Aliases do Unity Catalog Model Registry:**

O MLOps Stacks usa um sistema de aliases para controlar o ciclo de vida do modelo:

```
versão N → (sem alias)       ← recém treinado
versão N → "challenger"      ← passou na validação
versão N → "champion"        ← promovido para produção
```

---

### Passo 5: Deploy do Modelo {#passo-5-deploy-do-modelo}

**Notebook:** `deployment/model_deployment/notebooks/ModelDeployment.py`
**Script:** `deployment/model_deployment/deploy.py`

O deploy não significa necessariamente subir um endpoint REST — no contexto
batch, significa promover o modelo para o alias `champion`, tornando-o o
modelo "oficial" para inferência em batch.

**O que acontece:**

```python
def deploy(model_uri, env):
    # Obtém o model_uri da task anterior (Train)
    _, model_name, version = model_uri.split("/")
    client = MlflowClient(registry_uri="databricks-uc")

    # Atribui alias "champion" → este modelo será usado em produção
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version
    )

    # Remove "challenger" do modelo antigo
    client.delete_registered_model_alias(name=model_name, alias="challenger")
```

**Verificar o deploy:**

```bash
# Via CLI Databricks
databricks models get-version \
    --name "dev.validacao_dados_aula_8.validacao_dados_aula_8-model" \
    --version 1
```

Ou no Unity Catalog Explorer: **Catalog → dev → validacao_dados_aula_8 → Models**

---

### Passo 6: Inferência em Batch {#passo-6-inferência-em-batch}

**Notebook:** `deployment/batch_inference/notebooks/BatchInference.py`
**Script:** `deployment/batch_inference/predict.py`

Com o modelo `champion` disponível, podemos executar inferência em batch
sobre novos dados usando Spark — aproveitando a escala distribuída.

**O que acontece:**

```
taxi_scoring (Delta Table)
  trip_distance, num_passengers, hour_of_day, ...
                ↓
  mlflow.pyfunc.spark_udf()  ← carrega modelo como Spark UDF
                ↓
  predictions (Delta Table)
  trip_distance, ..., prediction, model_id, timestamp
```

**O script `predict.py`:**

```python
def predict_batch(spark_session, model_uri, input_table_name,
                  output_table_name, model_version, ts):
    mlflow.set_registry_uri("databricks-uc")

    # Carregar dados de scoring
    table = spark_session.table(input_table_name)

    # Criar Spark UDF a partir do modelo MLflow
    predict = mlflow.pyfunc.spark_udf(
        spark_session, model_uri, result_type="double",
        env_manager="virtualenv"
    )

    # Aplicar o modelo e adicionar metadados
    output_df = (
        table
        .withColumn("prediction", predict(struct(*table.columns)))
        .withColumn("model_id",   lit(model_version))
        .withColumn("timestamp",  to_timestamp(lit(ts)))
    )

    # Salvar resultados como Delta Table
    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)
```

**Executar o job de inferência:**

```bash
databricks bundle run batch_inference_job --target dev
```

**Resultado no Unity Catalog:**

```sql
-- Ver previsões geradas
SELECT * FROM dev.validacao_dados_aula_8.predictions
LIMIT 10;

-- Análise rápida das previsões
SELECT
  AVG(prediction)  AS avg_predicted_fare,
  MIN(prediction)  AS min_fare,
  MAX(prediction)  AS max_fare,
  COUNT(*)         AS total_records
FROM dev.validacao_dados_aula_8.predictions;
```

---

### Passo 7: Monitoramento {#passo-7-monitoramento}

**Notebook:** `monitoring/notebooks/MonitoredMetricViolationCheck.py`
**Script:** `monitoring/metric_violation_check_query.py`
**Config:** `resources/monitoring-resource.yml`

O monitoramento garante que o modelo continue performando bem em produção.
O Databricks Lakehouse Monitoring analisa continuamente as tabelas Delta
e gera perfis estatísticos automáticos.

**Conceito de Lakehouse Monitoring:**

```
predictions (Delta Table)
        ↓
Lakehouse Monitor (Databricks)
  ├── Profile Metrics Table (_profile_metrics)
  │   └── RMSE, drift, distribuição, etc.
  └── Drift Metrics Table (_drift_metrics)
        ↓
MetricViolationCheck (Notebook)
  ├── Verifica threshold nas últimas N janelas
  └── Se violado → aciona retraining
```

**A query de verificação (`metric_violation_check_query.py`):**

```sql
WITH recent_metrics AS (
  SELECT root_mean_squared_error, window
  FROM dev.validacao_dados_aula_8.predictions_profile_metrics
  WHERE column_name = ":table"
    AND slice_key IS NULL
    AND model_id != "*"
    AND log_type = "INPUT"
  ORDER BY window DESC
  LIMIT 5  -- últimas 5 janelas
)
SELECT CASE
  WHEN
    -- Pelo menos 2 das 5 janelas violam o threshold
    (SELECT COUNT(*) FROM recent_metrics
     WHERE root_mean_squared_error > 100) >= 2
    AND
    -- A janela mais recente também viola
    (SELECT root_mean_squared_error FROM recent_metrics
     ORDER BY window DESC LIMIT 1) > 100
  THEN 1  -- acionar retraining
  ELSE 0  -- tudo ok
END AS query_result
```

**Configuração do monitor (`resources/monitoring-resource.yml`):**

```yaml
resources:
  quality_monitors:
    predictions_monitor:
      table_name: ${var.catalog_name}.validacao_dados_aula_8.predictions
      assets_dir: /Shared/databricks_automl/validacao_dados_aula_8
      output_schema_name: ${var.catalog_name}.validacao_dados_aula_8
      inference_log:
        timestamp_col: timestamp
        granularities: ["1 day"]
        model_id_col: model_id
        prediction_col: prediction
        label_col: fare_amount
        problem_type: PROBLEM_TYPE_REGRESSION
      schedule:
        quartz_cron_expression: "0 0 8 * * ?"  # diariamente às 8h
```

---

## SAIBA MAIS — Databricks Asset Bundles {#saiba-mais}

### O que são os Databricks Asset Bundles?

**Databricks Asset Bundles (DABs)** — também chamados de *Declarative Automation Bundles* —
são a solução nativa da Databricks para definir, versionar e implantar recursos
do Databricks como **código declarativo** (Infrastructure as Code).

Com DABs, você descreve em arquivos YAML tudo o que seu projeto precisa:
jobs, pipelines Delta Live Tables, modelos MLflow, dashboards, secrets e mais.
A Databricks CLI então provisiona e gerencia esses recursos automaticamente
em seus workspaces de dev, staging e produção.

**Por que isso importa para MLOps?**

Sem DABs, cada workspace (dev, staging, prod) precisaria ser configurado manualmente:
criar jobs, definir parâmetros, configurar schedules, gerenciar permissões. Com DABs,
você define tudo uma única vez e faz deploy com um comando, garantindo consistência
entre ambientes e rastreabilidade via Git.

---

### Estrutura de um Bundle

O coração de um bundle é o arquivo `databricks.yml`, que fica na raiz do projeto:

```yaml
bundle:
  name: validacao_dados_aula_8   # nome do bundle

variables:                         # variáveis reutilizáveis
  catalog_name:
    description: "Catalog Unity Catalog para deploy"
    default: dev
  model_name:
    description: "Nome do modelo no registry"
    default: validacao_dados_aula_8-model

include:                           # arquivos de recursos incluídos
  - ./resources/model-workflow-resource.yml
  - ./resources/batch-inference-workflow-resource.yml
  - ./resources/ml-artifacts-resource.yml

targets:                           # ambientes de deploy
  dev:
    mode: development
    default: true
    variables:
      catalog_name: dev
    workspace:
      host: https://dev-workspace.databricks.com

  staging:
    variables:
      catalog_name: staging
    workspace:
      host: https://staging-workspace.databricks.com

  prod:
    variables:
      catalog_name: prod
    workspace:
      host: https://prod-workspace.databricks.com
```

**Elementos principais:**

| Elemento | Descrição |
|----------|-----------|
| `bundle.name` | Identificador único do bundle |
| `variables` | Valores parametrizáveis por ambiente |
| `include` | Arquivos YAML com definições de recursos |
| `targets` | Ambientes de deploy (dev/staging/prod) |
| `workspace.host` | URL do workspace Databricks alvo |

---

### Recursos Declarados como Código

Os recursos são definidos em arquivos YAML na pasta `resources/`. Cada arquivo
pode conter um ou mais recursos do tipo `jobs`, `pipelines`, `models`, etc.

**Exemplo: definição de um job de treinamento (`model-workflow-resource.yml`):**

```yaml
resources:
  jobs:
    model_training_job:
      name: ${bundle.target}-validacao_dados_aula_8-model-training-job

      # Cluster compartilhado para todas as tasks
      job_clusters:
        - job_cluster_key: training_cluster
          new_cluster:
            spark_version: 15.3.x-cpu-ml-scala2.12
            node_type_id: i3.xlarge
            num_workers: 2

      # Tasks do job (executadas em sequência)
      tasks:
        - task_key: Train
          job_cluster_key: training_cluster
          notebook_task:
            notebook_path: ../training/notebooks/Train.py
            base_parameters:
              env: ${bundle.target}
              training_data_path: ${var.catalog_name}.mlops_demo.taxi_fares_train
              model_name: ${var.catalog_name}.mlops_demo.${var.model_name}

        - task_key: ModelValidation
          depends_on:
            - task_key: Train          # executa APÓS Train
          notebook_task:
            notebook_path: ../validation/notebooks/ModelValidation.py
            base_parameters:
              run_mode: dry_run

        - task_key: ModelDeployment
          depends_on:
            - task_key: ModelValidation  # executa APÓS Validation
          notebook_task:
            notebook_path: ../deployment/model_deployment/notebooks/ModelDeployment.py

      # Agendamento: todos os dias às 9h UTC
      schedule:
        quartz_cron_expression: "0 0 9 * * ?"
        timezone_id: UTC

      # Permissões: todos os usuários podem visualizar
      permissions:
        - level: CAN_VIEW
          group_name: users
```

**Exemplo: definição do modelo MLflow (`ml-artifacts-resource.yml`):**

```yaml
resources:
  experiments:
    mlflow_experiment:
      name: ${var.experiment_name}
      permissions:
        - level: CAN_READ
          group_name: users

  registered_models:
    mlflow_registered_model:
      name: ${var.catalog_name}.validacao_dados_aula_8.${var.model_name}
      catalog_name: ${var.catalog_name}
      schema_name: validacao_dados_aula_8
      comment: "Modelo de previsão de tarifa de táxi — criado pelo MLOps Stacks"
      grants:
        - principal: account users
          privileges:
            - EXECUTE
            - APPLY_TAG
```

---

### Variáveis e Substituição

O DAB suporta substituição de variáveis em tempo de deploy usando a sintaxe
`${var.nome}` e `${bundle.alvo}`:

```yaml
# Definição
variables:
  catalog_name:
    default: dev

# Uso
model_name: ${var.catalog_name}.schema.modelo   # → "dev.schema.modelo"
job_name: ${bundle.target}-meu-job              # → "staging-meu-job"
user: ${workspace.current_user.userName}         # → "email@empresa.com"
```

Isso garante que o mesmo código funcione em qualquer ambiente, apenas alterando
o `--target` no comando de deploy.

---

### Comandos Essenciais da CLI

```bash
# 1. Instalar a Databricks CLI
pip install databricks-cli

# 2. Configurar autenticação
databricks configure --token
# ou com variável de ambiente
export DATABRICKS_HOST=https://workspace.databricks.com
export DATABRICKS_TOKEN=dapiXXXXXXXX

# 3. Validar a configuração do bundle (sem fazer deploy)
#    Detecta erros de sintaxe e referências inválidas
databricks bundle validate

# 4. Gerar o bundle sem fazer deploy (útil para debug)
databricks bundle generate

# 5. Fazer deploy para o ambiente DEV (padrão)
databricks bundle deploy

# 6. Fazer deploy para staging
databricks bundle deploy --target staging

# 7. Fazer deploy para produção
databricks bundle deploy --target prod

# 8. Executar um job manualmente após o deploy
databricks bundle run model_training_job --target dev

# 9. Ver o status dos jobs deployados
databricks jobs list

# 10. Destruir todos os recursos do bundle (cuidado!)
databricks bundle destroy --target dev
```

---

### O Modo `development`

O target `dev` tem `mode: development`, que ativa comportamentos especiais:

```yaml
targets:
  dev:
    mode: development   # ← ativa o modo dev
```

No modo `development`:
- Jobs são nomeados com prefixo `[dev username]` para evitar conflitos
- Schedules são pausados automaticamente
- Experimentos MLflow são criados na pasta pessoal do usuário
- Ideal para desenvolvimento e testes sem afetar outros membros do time

---

### Ciclo de Vida com CI/CD (GitHub Actions)

O MLOps Stacks inclui workflows GitHub Actions pré-configurados para CI/CD:

```
Developer faz push/PR para main
         ↓
.github/workflows/validacao_dados_aula_8-bundle-ci.yml
  ├── Executa testes unitários (pytest)
  ├── Valida o bundle (databricks bundle validate)
  └── Faz deploy para staging (databricks bundle deploy --target staging)
         ↓
Merge para branch release
         ↓
.github/workflows/validacao_dados_aula_8-bundle-cd-prod.yml
  └── Faz deploy para produção (databricks bundle deploy --target prod)
```

**Como funciona na prática:**

1. Data Scientist modifica o `Train.py` (ex: novos hiperparâmetros)
2. Abre um PR com as mudanças
3. CI executa automaticamente: testes, validação do bundle, deploy em staging
4. Se tudo passa, o PR é mergeado e o código vai para produção via CD

---

### DABs no Contexto do MLOps Stacks

O MLOps Stacks é essencialmente um template de DAB especializado para Machine Learning.
A relação entre os componentes é:

```
databricks.yml (bundle raiz)
  │
  ├── resources/model-workflow-resource.yml
  │     └── Job: Train → Validate → Deploy
  │           (schedule: diário às 9h)
  │
  ├── resources/batch-inference-workflow-resource.yml
  │     └── Job: BatchInference
  │           (schedule: diário às 11h)
  │
  ├── resources/ml-artifacts-resource.yml
  │     ├── MLflow Experiment
  │     └── Registered Model (Unity Catalog)
  │
  └── resources/monitoring-resource.yml
        ├── Lakehouse Monitor
        ├── Job: MetricViolationCheck
        └── Job: RetrainingTrigger
```

**Diagrama de deploy por ambiente:**

```
Git Repository
     │
     ├──── databricks bundle deploy --target dev
     │           ↓
     │     DEV Workspace
     │     [dev username]-model-training-job
     │     dev.validacao_dados_aula_8.model (Unity Catalog)
     │
     ├──── databricks bundle deploy --target staging
     │           ↓
     │     STAGING Workspace
     │     staging-model-training-job
     │     staging.validacao_dados_aula_8.model
     │
     └──── databricks bundle deploy --target prod
                 ↓
           PROD Workspace
           prod-model-training-job
           prod.validacao_dados_aula_8.model
```

---

### Referências

- [Documentação Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles)
- [MLOps Stacks no GitHub](https://github.com/databricks/mlops-stacks)
- [Documentação MLOps Stacks](https://docs.databricks.com/machine-learning/mlops/mlops-stacks.html)
- [Schema do databricks.yml](https://docs.databricks.com/dev-tools/bundles/reference.html)
- [mlflow.evaluate() API](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate)
- [Unity Catalog Model Registry](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
