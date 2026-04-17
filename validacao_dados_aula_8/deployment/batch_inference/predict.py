import mlflow
from pyspark.sql import functions as F
from pyspark.sql.functions import struct, lit, to_timestamp


def _add_features(df):
    return (
        df
        .withColumn("is_rush_hour",
            F.when(((F.col("hour_of_day") >= 7) & (F.col("hour_of_day") <= 9)) |
                   ((F.col("hour_of_day") >= 17) & (F.col("hour_of_day") <= 19)), 1).otherwise(0))
        .withColumn("is_weekend",
            F.when(F.col("day_of_week") >= 5, 1).otherwise(0))
        .withColumn("is_bad_weather",
            F.when(F.col("weather_score") < 3.0, 1).otherwise(0))
        .withColumn("distance_x_passengers",
            F.col("trip_distance") * F.col("num_passengers"))
    )


def predict_batch(
    spark_session, model_uri, input_table_name, output_table_name, model_version, ts
):
    """
    Apply the model at the specified URI for batch inference on the table with name input_table_name,
    writing results to the table with name output_table_name
    """

    mlflow.set_registry_uri("databricks-uc")

    table = _add_features(spark_session.table(input_table_name))

    predict = mlflow.pyfunc.spark_udf(
        spark_session, model_uri, result_type="double", env_manager="local"
    )

    output_df = (
        table.withColumn("prediction", predict(struct(*table.columns)))
        .withColumn("model_id", lit(model_version))
        .withColumn("timestamp", to_timestamp(lit(ts)))
    )
    output_df.display()

    # Model predictions are written to the Delta table provided as input.
    # Delta is the default format in Databricks Runtime 8.0 and above.
    output_df.write.format("delta").mode("overwrite").saveAsTable(output_table_name)