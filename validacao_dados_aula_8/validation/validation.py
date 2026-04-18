"""
Módulo de validação do modelo — funções de métricas customizadas e thresholds.

Este módulo é chamado pelo notebook ModelValidation.py através de importação dinâmica.
Edite as funções abaixo para definir as regras de qualidade do modelo.

Documentação:
  - mlflow.evaluate: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.evaluate
  - Model Validation: https://mlflow.org/docs/latest/models.html#model-validation
  - Pandera: https://pandera.readthedocs.io/
"""

import numpy as np
from mlflow.models import make_metric, MetricThreshold


# ---------------------------------------------------------------------------
# 1. Métricas customizadas
# ---------------------------------------------------------------------------

def custom_metrics():
    """
    Define métricas customizadas para avaliação do modelo.

    Além das métricas built-in do MLflow (rmse, mae, r2_score, etc.),
    podemos adicionar nossas próprias métricas de negócio.

    Retorne [] se não precisar de métricas customizadas.
    """

    def mean_absolute_percentage_error(eval_df, _builtin_metrics):
        """
        MAPE — útil para entender o erro relativo em percentual.
        Ex: MAPE de 0.10 significa que o modelo erra, em média, 10% do valor real.

        Nota: evitar usar quando target pode ser zero.
        """
        y_true = eval_df["target"]
        y_pred = eval_df["prediction"]
        # Evitar divisão por zero
        mask   = y_true != 0
        mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        return float(mape)

    def large_error_rate(eval_df, _builtin_metrics):
        """
        Fração de previsões com erro absoluto > $5.00.

        Métrica de negócio: queremos menos de 20% de erros "grandes"
        (definidos como erros maiores que $5 no contexto de corridas de táxi).
        """
        abs_error   = np.abs(eval_df["prediction"] - eval_df["target"])
        large_errors = (abs_error > 5.0).sum()
        return float(large_errors / len(eval_df))

    return [
        make_metric(eval_fn=mean_absolute_percentage_error, greater_is_better=False, name="mape"),
        make_metric(eval_fn=large_error_rate,               greater_is_better=False, name="large_error_rate"),
    ]


# ---------------------------------------------------------------------------
# 2. Thresholds de validação
# ---------------------------------------------------------------------------

def validation_thresholds():
    """
    Define os critérios mínimos de qualidade para que o modelo seja promovido.

    Se qualquer threshold for violado em run_mode="enabled", o deployment é bloqueado.
    Em run_mode="dry_run", as falhas são logadas mas não bloqueiam o deploy.

    Métricas built-in disponíveis para regressão:
      - mean_squared_error (MSE)
      - root_mean_squared_error (RMSE)
      - mean_absolute_error (MAE)
      - max_error
      - r2_score

    Thresholds configurados para o dataset sintético de táxi (valores em USD):
      - RMSE   ≤ 4.00  (erro médio quadrático aceitável para corridas de ~$15 médios)
      - MAE    ≤ 3.00  (erro absoluto médio)
      - R²     ≥ 0.85  (modelo explica pelo menos 85% da variância)
      - MAPE   ≤ 0.25  (erro relativo ≤ 25%)
      - large_error_rate ≤ 0.20 (max 20% de previsões com erro > $5)
    """
    return {
        # Métricas built-in do MLflow
        "root_mean_squared_error": MetricThreshold(
            threshold=4.00,         # RMSE deve ser ≤ $4.00
            greater_is_better=False,
        ),
        "mean_absolute_error": MetricThreshold(
            threshold=3.00,         # MAE deve ser ≤ $3.00
            greater_is_better=False,
        ),
        "r2_score": MetricThreshold(
            threshold=0.85,         # R² deve ser ≥ 0.85
            greater_is_better=True,
        ),
        # Métricas customizadas definidas em custom_metrics()
        "mape": MetricThreshold(
            threshold=0.25,         # MAPE ≤ 25%
            greater_is_better=False,
        ),
        "large_error_rate": MetricThreshold(
            threshold=0.20,         # ≤ 20% de erros grandes
            greater_is_better=False,
        ),
    }


# ---------------------------------------------------------------------------
# 3. Configuração do avaliador
# ---------------------------------------------------------------------------

def evaluator_config():
    """
    Configurações adicionais para o mlflow.evaluate().

    Pode ser usado para:
      - Especificar colunas a ignorar
      - Configurar o evaluator "default" do MLflow

    Retorne {} se não precisar de configurações extras.
    """
    return {
        # Colunas que não são features (serão ignoradas pelo evaluator)
        # O MLflow já lida com isso via 'targets', mas podemos ser explícitos
    }


# ---------------------------------------------------------------------------
# 4. Schema de entrada — Pandera
# ---------------------------------------------------------------------------

def pandera_schema():
    """
    Define o schema Pandera para validação do dataset de entrada.

    Integração complementar ao mlflow.evaluate(): enquanto o MLflow valida
    a qualidade das métricas do modelo, o Pandera valida a integridade e
    conformidade dos dados de entrada antes da avaliação.

    Tipos, ranges e ausência de nulos são verificados para todas as colunas
    originais do dataset de corridas de táxi.

    Documentação: https://pandera.readthedocs.io/
    """
    import pandera as pa

    return pa.DataFrameSchema(
        columns={
            "trip_distance":  pa.Column(float, pa.Check.between(0.0, 50.0),  nullable=False,
                                        description="Distância da corrida em km (0.5–30 km)"),
            "num_passengers": pa.Column(int,   pa.Check.between(1, 6),       nullable=False,
                                        description="Número de passageiros (1–6)"),
            "hour_of_day":    pa.Column(int,   pa.Check.between(0, 23),      nullable=False,
                                        description="Hora de início da corrida (0–23)"),
            "day_of_week":    pa.Column(int,   pa.Check.between(0, 6),       nullable=False,
                                        description="Dia da semana (0=seg, 6=dom)"),
            "weather_score":  pa.Column(float, pa.Check.between(0.0, 10.0), nullable=False,
                                        description="Índice climático (0=péssimo, 10=ótimo)"),
            "pickup_zone":    pa.Column(int,   pa.Check.between(1, 5),       nullable=False,
                                        description="Zona de origem (1–5)"),
            "dropoff_zone":   pa.Column(int,   pa.Check.between(1, 5),       nullable=False,
                                        description="Zona de destino (1–5)"),
            "fare_amount":    pa.Column(float, pa.Check.greater_than(0.0),   nullable=False,
                                        description="Valor da corrida em USD (target)"),
        },
        coerce=True,
    )
