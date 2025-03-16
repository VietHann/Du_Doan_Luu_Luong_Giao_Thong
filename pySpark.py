
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import StringIndexer

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import os
import joblib


warnings.filterwarnings('ignore')


spark = SparkSession.builder.appName("TransportPrediction").getOrCreate()
print("Spark version:", spark.version)
print("Libraries imported successfully!")




print("Reading data from CSV file...")
try:

    file_path = "/dbfs/FileStore/data/mta_1710.csv"


    df = spark.read.csv(file_path, header=True, inferSchema=True, mode="DROPMALFORMED")
    print(f"Number of records: {df.count()}")
    print(f"Data columns: {df.columns}")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    raise


for column in df.columns:
    if "." in column:
        new_column = column.replace(".", "")
        df = df.withColumnRenamed(column, new_column)


null_counts = {}
for column in df.columns:
    null_count = df.filter(df[column].isNull()).count()
    null_counts[column] = null_count

print("Null values in dataset:")
for column, count in null_counts.items():
    print(f"{column}: {count}")


numeric_cols = [
    "OriginLat", "OriginLong",
    "VehicleLocationLatitude", "VehicleLocationLongitude",
    "DestinationLat", "DestinationLong",
    "DistanceFromStop"
]

important_cols = ["OriginName", "DestinationName", "NextStopPointName",
                 "VehicleLocationLatitude", "VehicleLocationLongitude", "DistanceFromStop"]


for col_name in numeric_cols:
    if col_name in df.columns:
        df = df.withColumn(col_name, df[col_name].cast("float"))


df_clean = df.dropna(subset=important_cols)
print(f"Number of records after removing null values: {df_clean.count()}")




df_clean = df_clean.withColumn("RecordedAtTime", F.to_timestamp("RecordedAtTime"))


df_clean = df_clean.withColumn("Hour", F.hour("RecordedAtTime"))
df_clean = df_clean.withColumn("DayOfWeek", F.dayofweek("RecordedAtTime"))
df_clean = df_clean.withColumn("Month", F.month("RecordedAtTime"))
df_clean = df_clean.withColumn("WeekOfYear", F.weekofyear("RecordedAtTime"))


df_clean = df_clean.withColumn("DayOfMonth", F.dayofmonth("RecordedAtTime"))
df_clean = df_clean.withColumn("DayOfYear", F.dayofyear("RecordedAtTime"))


df_clean = df_clean.withColumn("IsMonthEnd",
                              F.when((F.dayofmonth("RecordedAtTime") >= 28) &
                                    (F.dayofmonth("RecordedAtTime") <= 31), 1).otherwise(0))
df_clean = df_clean.withColumn("IsMonthStart",
                              F.when(F.dayofmonth("RecordedAtTime") == 1, 1).otherwise(0))


df_clean = df_clean.withColumn("IsMorningPeak",
                              F.when((F.col("Hour") >= 7) & (F.col("Hour") <= 9), 1).otherwise(0))
df_clean = df_clean.withColumn("IsEveningPeak",
                              F.when((F.col("Hour") >= 16) & (F.col("Hour") <= 18), 1).otherwise(0))
df_clean = df_clean.withColumn("IsWeekend",
                              F.when((F.col("DayOfWeek") == 1) | (F.col("DayOfWeek") == 7), 1).otherwise(0))


df_clean = df_clean.withColumn("Hour_sin", F.sin(2 * np.pi * F.col("Hour")/24))
df_clean = df_clean.withColumn("Hour_cos", F.cos(2 * np.pi * F.col("Hour")/24))
df_clean = df_clean.withColumn("DayOfWeek_sin", F.sin(2 * np.pi * F.col("DayOfWeek")/7))
df_clean = df_clean.withColumn("DayOfWeek_cos", F.cos(2 * np.pi * F.col("DayOfWeek")/7))
df_clean = df_clean.withColumn("Month_sin", F.sin(2 * np.pi * F.col("Month")/12))
df_clean = df_clean.withColumn("Month_cos", F.cos(2 * np.pi * F.col("Month")/12))


numeric_features = [
    "OriginLat", "OriginLong",
    "DestinationLat", "DestinationLong",
    "VehicleLocationLatitude", "VehicleLocationLongitude",
    "Hour", "DayOfWeek", "Month", "WeekOfYear",
    "IsMorningPeak", "IsEveningPeak", "IsWeekend",
    "DayOfMonth", "DayOfYear", "IsMonthEnd", "IsMonthStart",
    "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos"
]

categorical_cols = [
    "PublishedLineName",
    "OriginName",
    "DestinationName",
    "NextStopPointName",
    "DirectionRef"
]


available_features = [col for col in numeric_features if col in df_clean.columns]
print(f"Using features: {available_features}")


for col in available_features:

    mean_val = df_clean.select(F.mean(F.col(col))).collect()[0][0]

    df_clean = df_clean.withColumn(col, F.coalesce(F.col(col), F.lit(mean_val)))


df_clean = df_clean.withColumn("timestamp", df_clean["RecordedAtTime"].cast("long"))
df_clean = df_clean.orderBy("timestamp")


total_rows = df_clean.count()
train_size = int(0.7 * total_rows)
test_size = int(0.2 * total_rows)


train_df = df_clean.limit(train_size)
test_df = df_clean.filter(F.row_number().over(F.Window.orderBy("timestamp")) > train_size) \
                  .filter(F.row_number().over(F.Window.orderBy("timestamp")) <= (train_size + test_size))
val_df = df_clean.filter(F.row_number().over(F.Window.orderBy("timestamp")) > (train_size + test_size))



assembler = VectorAssembler(inputCols=available_features, outputCol="features")


scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)


pipeline = Pipeline(stages=[assembler, scaler])


fitted_pipeline = pipeline.fit(train_df)


train_data = fitted_pipeline.transform(train_df)
test_data = fitted_pipeline.transform(test_df)
val_data = fitted_pipeline.transform(val_df)

print(f"Training samples: {train_data.count()}")
print(f"Test samples: {test_data.count()}")
print(f"Validation samples: {val_data.count()}")




def calculate_metrics(predictions_df, label_col="DistanceFromStop", prediction_col="prediction"):

    pred_pd = predictions_df.select(label_col, prediction_col).toPandas()

    y_true = pred_pd[label_col]
    y_pred = pred_pd[prediction_col]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)


    bins = np.linspace(min(y_true), max(y_true), 5)
    y_true_binned = np.digitize(y_true, bins)
    y_pred_binned = np.digitize(y_pred, bins)

    cm = confusion_matrix(y_true_binned, y_pred_binned)

    return rmse, r2, cm, bins


evaluator = RegressionEvaluator(labelCol="DistanceFromStop", predictionCol="prediction", metricName="rmse")



lr = LinearRegression(featuresCol="scaledFeatures", labelCol="DistanceFromStop", maxIter=50, regParam=0.3, elasticNetParam=0.8)


rf_50 = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="DistanceFromStop",
                            numTrees=50, maxDepth=10, seed=42)

rf_100 = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="DistanceFromStop",
                             numTrees=100, maxDepth=15, seed=42)

rf_200 = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="DistanceFromStop",
                             numTrees=200, maxDepth=15, seed=42)

rf_ts = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="DistanceFromStop",
                            numTrees=100, maxDepth=15, seed=42,
                            featureSubsetStrategy="sqrt", minInstancesPerNode=4)


models = {
    "LinearRegression": lr,
    "RandomForest_50": rf_50,
    "RandomForest_100": rf_100,
    "RandomForest_200": rf_200,
    "TimeSeries_RF": rf_ts
}


from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol="scaledFeatures", labelCol="DistanceFromStop",
                  maxIter=100, maxDepth=6, stepSize=0.1, seed=42)
models["GBTRegressor"] = gbt


results = {}
all_models = {}
confusion_matrices = {}

print("Training and evaluating models...")
for name, model in models.items():
    start_time = time.time()
    try:
        print(f"Training {name} model...")


        fitted_model = model.fit(train_data)


        train_predictions = fitted_model.transform(train_data)
        test_predictions = fitted_model.transform(test_data)
        val_predictions = fitted_model.transform(val_data)


        train_rmse, train_r2, train_cm, _ = calculate_metrics(train_predictions)
        test_rmse, test_r2, test_cm, cm_bins = calculate_metrics(test_predictions)
        val_rmse, val_r2, val_cm, _ = calculate_metrics(val_predictions)


        training_time = time.time() - start_time


        results[name] = {
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "training_time": training_time,
            "predictions": {
                "train": train_predictions,
                "test": test_predictions,
                "val": val_predictions
            }
        }

        all_models[name] = fitted_model
        confusion_matrices[name] = (test_cm, cm_bins)

        print(f"  Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        print(f"  Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        print(f"  Training time: {training_time:.2f} seconds")

    except Exception as e:
        print(f"Error training {name} model: {e}")
        continue




if len(results) >= 3:
    try:
        print("\nCreating ensemble of best models...")


        sorted_models = sorted(results.items(), key=lambda x: x[1]["test_rmse"])
        top_3_models = [name for name, _ in sorted_models[:3]]

        print(f"Top 3 models for ensemble: {top_3_models}")



        ensemble_data = {
            "train": pd.DataFrame(),
            "test": pd.DataFrame(),
            "val": pd.DataFrame()
        }


        for name in top_3_models:
            for dataset in ["train", "test", "val"]:
                predictions_df = results[name]["predictions"][dataset]

                model_preds = predictions_df.select("DistanceFromStop", "prediction").toPandas()

                if dataset == "test" and ensemble_data[dataset].empty:
                    ensemble_data[dataset]["DistanceFromStop"] = model_preds["DistanceFromStop"]

                ensemble_data[dataset][f"{name}_pred"] = model_preds["prediction"]


        for dataset in ["train", "test", "val"]:
            pred_columns = [f"{name}_pred" for name in top_3_models]
            ensemble_data[dataset]["ensemble_pred"] = ensemble_data[dataset][pred_columns].mean(axis=1)


        test_preds = ensemble_data["test"]
        ensemble_test_rmse = np.sqrt(mean_squared_error(test_preds["DistanceFromStop"], test_preds["ensemble_pred"]))
        ensemble_test_r2 = r2_score(test_preds["DistanceFromStop"], test_preds["ensemble_pred"])


        bins = np.linspace(min(test_preds["DistanceFromStop"]), max(test_preds["DistanceFromStop"]), 5)
        y_true_binned = np.digitize(test_preds["DistanceFromStop"], bins)
        y_pred_binned = np.digitize(test_preds["ensemble_pred"], bins)
        ensemble_test_cm = confusion_matrix(y_true_binned, y_pred_binned)


        train_preds = ensemble_data["train"]
        val_preds = ensemble_data["val"]

        ensemble_train_rmse = np.sqrt(mean_squared_error(train_preds["DistanceFromStop"], train_preds["ensemble_pred"]))
        ensemble_train_r2 = r2_score(train_preds["DistanceFromStop"], train_preds["ensemble_pred"])

        ensemble_val_rmse = np.sqrt(mean_squared_error(val_preds["DistanceFromStop"], val_preds["ensemble_pred"]))
        ensemble_val_r2 = r2_score(val_preds["DistanceFromStop"], val_preds["ensemble_pred"])


        results["Ensemble_Top3"] = {
            "train_rmse": ensemble_train_rmse,
            "train_r2": ensemble_train_r2,
            "test_rmse": ensemble_test_rmse,
            "test_r2": ensemble_test_r2,
            "val_rmse": ensemble_val_rmse,
            "val_r2": ensemble_val_r2,
            "training_time": sum([results[name]["training_time"] for name in top_3_models]),
            "component_models": top_3_models
        }

        confusion_matrices["Ensemble_Top3"] = (ensemble_test_cm, bins)

        print(f"  Ensemble Train RMSE: {ensemble_train_rmse:.4f}, R²: {ensemble_train_r2:.4f}")
        print(f"  Ensemble Test RMSE: {ensemble_test_rmse:.4f}, R²: {ensemble_test_r2:.4f}")
        print(f"  Ensemble Validation RMSE: {ensemble_val_rmse:.4f}, R²: {ensemble_val_r2:.4f}")

    except Exception as e:
        print(f"Error creating ensemble: {e}")


if results:

    best_model_name = min(results, key=lambda x: results[x]["test_rmse"])
    if best_model_name == "Ensemble_Top3":
        print("\nBest model is Ensemble of:", results[best_model_name]["component_models"])
        best_model = None
    else:
        best_model = all_models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")

    if best_model_name != "Ensemble_Top3":
        print(f"Training time: {results[best_model_name]['training_time']:.2f} seconds")


    if best_model is not None:
        try:

            model_dir = "/dbfs/FileStore/models"
            os.makedirs(model_dir, exist_ok=True)


            import mlflow

            experiment_name = "/Transport-Prediction"


            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{best_model_name}_model"):

                if hasattr(best_model, "getNumTrees"):
                    mlflow.log_param("num_trees", best_model.getNumTrees())
                if hasattr(best_model, "getMaxDepth"):
                    mlflow.log_param("max_depth", best_model.getMaxDepth())


                mlflow.log_metric("rmse", results[best_model_name]["test_rmse"])
                mlflow.log_metric("r2", results[best_model_name]["test_r2"])


                mlflow.spark.log_model(best_model, "spark-model")


                mlflow.spark.log_model(fitted_pipeline, "pipeline")

                print(f"Model and pipeline saved to MLflow")

        except Exception as e:
            print(f"Could not save model: {e}")


    print("\nSummary of model results:")
    print("Model                Train RMSE    Test RMSE     Val RMSE     R²        Time(s)")
    print("-" * 80)
    for name, result in results.items():
        print(f"{name:<20} {result['train_rmse']:.4f}      {result['test_rmse']:.4f}      " +
              f"{result['val_rmse']:.4f}      {result['test_r2']:.4f}    {result.get('training_time', 0):.2f}")


    viz_dir = "/dbfs/FileStore/visualizations"
    os.makedirs(viz_dir, exist_ok=True)


    plt.figure(figsize=(14, 7))
    model_names = list(results.keys())
    train_rmse = [results[name]['train_rmse'] for name in model_names]
    test_rmse = [results[name]['test_rmse'] for name in model_names]
    val_rmse = [results[name]['val_rmse'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.bar(x - width, train_rmse, width, label='Train RMSE')
    plt.bar(x, test_rmse, width, label='Test RMSE')
    plt.bar(x + width, val_rmse, width, label='Validation RMSE')

    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison Across Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/rmse_comparison.png")
    plt.close()


    plt.figure(figsize=(14, 7))
    train_r2 = [results[name]['train_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    val_r2 = [results[name]['val_r2'] for name in model_names]

    plt.bar(x - width, train_r2, width, label='Train R²')
    plt.bar(x, test_r2, width, label='Test R²')
    plt.bar(x + width, val_r2, width, label='Validation R²')

    plt.xlabel('Models')
    plt.ylabel('R²')
    plt.title('R² Comparison Across Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/r2_comparison.png")
    plt.close()


    plt.figure(figsize=(14, 7))
    times = [results[name].get('training_time', 0) for name in model_names]

    plt.bar(x, times, 0.4)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/training_time_comparison.png")
    plt.close()


    if best_model_name.startswith("RandomForest") or best_model_name == "GBTRegressor":

        feature_importances = best_model.featureImportances.toArray()



        feature_names = available_features


        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(feature_importances)],
            'Importance': feature_importances
        })


        importance_df = importance_df.sort_values('Importance', ascending=False)


        top_n = min(20, len(importance_df))
        importance_df = importance_df.head(top_n)

        plt.figure(figsize=(14, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'{best_model_name} - Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/{best_model_name}_feature_importance.png")
        plt.close()


    if best_model_name != "Ensemble_Top3" and best_model_name in results:

        test_predictions_df = results[best_model_name]["predictions"]["test"]
        test_true_pred = test_predictions_df.select("DistanceFromStop", "prediction").toPandas()

        y_true = test_true_pred["DistanceFromStop"]
        y_pred = test_true_pred["prediction"]

        plt.figure(figsize=(14, 10))


        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{best_model_name}: Actual vs Predicted')


        plt.subplot(2, 2, 2)
        residuals = y_pred - y_true
        plt.hist(residuals, bins=50)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')


        plt.subplot(2, 2, 3)
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')


        from scipy import stats
        plt.subplot(2, 2, 4)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')

        plt.tight_layout()
        plt.savefig(f"{viz_dir}/residual_analysis.png")
        plt.close()

    print(f"Visualizations saved to {viz_dir}")
else:
    print("No models were successfully trained for comparison")

print("\nAnalysis completed successfully!")




if best_model_name.startswith("RandomForest") or best_model_name == "GBTRegressor":
    print("Performing detailed feature analysis for best model...")


    feature_importances = best_model.featureImportances.toArray()


    feature_names = available_features


    feature_importance = pd.DataFrame({
        'Feature': feature_names[:len(feature_importances)],
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)


    top_10_features = feature_importance['Feature'].head(10).tolist()

    print(f"Top 10 most important features: {top_10_features}")




    test_df_pd = test_data.select(["DistanceFromStop"] + top_10_features).toPandas()


    plt.figure(figsize=(15, 12))
    for i, feature in enumerate(top_10_features[:5]):
        plt.subplot(2, 3, i+1)


        plt.scatter(test_df_pd[feature], test_df_pd["DistanceFromStop"], alpha=0.3, s=10)


        try:
            z = np.polyfit(test_df_pd[feature], test_df_pd["DistanceFromStop"], 1)
            p = np.poly1d(z)
            plt.plot(test_df_pd[feature], p(test_df_pd[feature]), "r--")
        except:
            print(f"Could not fit trend line for {feature}")

        plt.xlabel(feature)
        plt.ylabel('DistanceFromStop')
        plt.title(f'Impact of {feature}')

    plt.tight_layout()
    plt.savefig("/dbfs/FileStore/visualizations/top_features_dependence.png")
    plt.close()


    plt.figure(figsize=(12, 10))


    corr_matrix = test_df_pd[top_10_features].corr()


    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation between top 10 important features')
    plt.tight_layout()
    plt.savefig("/dbfs/FileStore/visualizations/feature_correlation.png")
    plt.close()


    plt.figure(figsize=(12, 8))

    importance_sorted = feature_importance.sort_values('Importance', ascending=False)
    plt.barh(importance_sorted['Feature'], importance_sorted['Importance'])
    plt.xlabel('Feature Importance')
    plt.title(f'{best_model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig("/dbfs/FileStore/visualizations/model_feature_importance_detailed.png")
    plt.close()






    top_3_features = top_10_features[:3]

    for feature in top_3_features:

        min_val = test_df_pd[feature].min()
        max_val = test_df_pd[feature].max()
        grid_points = np.linspace(min_val, max_val, 20)


        avg_predictions = []


        for point in grid_points:

            modified_df = test_df_pd.copy()

            modified_df[feature] = point


            modified_spark_df = spark.createDataFrame(modified_df)

            modified_assembled = assembler.transform(modified_spark_df)
            modified_scaled = scaler.transform(modified_assembled)


            predictions = best_model.transform(modified_scaled)


            avg_pred = predictions.select(F.avg("prediction")).collect()[0][0]
            avg_predictions.append(avg_pred)


        plt.figure(figsize=(10, 6))
        plt.plot(grid_points, avg_predictions)
        plt.xlabel(feature)
        plt.ylabel('Average Prediction (DistanceFromStop)')
        plt.title(f'Partial Dependence Plot for {feature}')
        plt.grid(True)
        plt.savefig(f"/dbfs/FileStore/visualizations/pdp_{feature.replace('.', '_')}.png")
        plt.close()


if best_model_name != "Ensemble_Top3" and best_model_name in results:
    try:

        time_pred_df = results[best_model_name]["predictions"]["test"].select(
            "RecordedAtTime", "DistanceFromStop", "prediction"
        ).toPandas()


        time_pred_df = time_pred_df.sort_values("RecordedAtTime")


        time_pred_df["error"] = time_pred_df["prediction"] - time_pred_df["DistanceFromStop"]


        time_pred_df.set_index("RecordedAtTime", inplace=True)
        daily_data = time_pred_df.resample('D').mean()


        plt.figure(figsize=(15, 8))
        plt.plot(daily_data.index, daily_data["DistanceFromStop"], label="Actual", marker='o', markersize=4)
        plt.plot(daily_data.index, daily_data["prediction"], label="Predicted", marker='x', markersize=4)
        plt.xlabel("Date")
        plt.ylabel("DistanceFromStop")
        plt.title(f"{best_model_name}: Actual vs Predicted Values Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/dbfs/FileStore/visualizations/predictions_over_time.png")
        plt.close()


        plt.figure(figsize=(15, 8))
        plt.plot(daily_data.index, daily_data["error"], color='red')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Prediction Error")
        plt.title(f"{best_model_name}: Prediction Error Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("/dbfs/FileStore/visualizations/prediction_error_over_time.png")
        plt.close()


        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        plt.figure(figsize=(12, 6))
        plot_acf(daily_data["error"].dropna(), lags=30, title=f"Autocorrelation of Prediction Errors")
        plt.tight_layout()
        plt.savefig("/dbfs/FileStore/visualizations/error_autocorrelation.png")
        plt.close()

    except Exception as e:
        print(f"Error in time series analysis: {e}")


print("Complete analysis completed!")
print("Visualizations saved to /dbfs/FileStore/visualizations/")


if best_model_name != "Ensemble_Top3" and best_model_name in results:

    test_predictions = results[best_model_name]["predictions"]["test"]
    test_predictions_df = test_predictions.select("RecordedAtTime", "DistanceFromStop", "prediction")


    test_predictions_df.write.mode("overwrite").option("header", "true").csv("/dbfs/FileStore/results/test_predictions.csv")
    print("Test predictions saved to /dbfs/FileStore/results/test_predictions.csv")


    evaluation_data = []
    for name, result in results.items():
        evaluation_data.append({
            "Model": name,
            "Train_RMSE": result['train_rmse'],
            "Test_RMSE": result['test_rmse'],
            "Val_RMSE": result['val_rmse'],
            "Test_R2": result['test_r2'],
            "Training_Time": result.get('training_time', 0)
        })


    eval_df = spark.createDataFrame(evaluation_data)
    eval_df.write.mode("overwrite").option("header", "true").csv("/dbfs/FileStore/results/model_evaluation.csv")
    print("Model evaluation metrics saved to /dbfs/FileStore/results/model_evaluation.csv")

print("Analysis and export completed successfully!")
