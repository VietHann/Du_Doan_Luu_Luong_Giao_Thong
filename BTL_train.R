library(tidyverse)
library(lubridate)

library(caret)
library(xgboost)
library(randomForest)


library(Metrics)
library(yardstick)



gpu_available <- tryCatch({



  library(gpuR)
  gpuMatrix(1:10) %*% gpuMatrix(1:10)
  TRUE
}, error = function(e) {
  FALSE
})

if (gpu_available) {
  print("GPU có vẻ khả dụng.")
} else {
  print("GPU không khả dụng hoặc chưa được thiết lập.")
}

print("Các thư viện đã được import thành công!")




print("Đọc dữ liệu từ file CSV...")
try({

  df <- read.csv("/dbfs/FileStore/tables/mta_1710.csv", stringsAsFactors = FALSE)
  print(paste("Số lượng bản ghi:", nrow(df)))
  print("Tên các cột:")
  print(names(df))
}, error = function(e) {
  print(paste("Lỗi khi đọc file CSV:", e))
  stop(e)
})


names(df) <- gsub("VehicleLocation.Latitude", "VehicleLocationLatitude", names(df))
names(df) <- gsub("VehicleLocation.Longitude", "VehicleLocationLongitude", names(df))


print("Số lượng giá trị NULL trong mỗi cột:")
print(colSums(is.na(df)))


numeric_cols <- c(
  "OriginLat", "OriginLong",
  "VehicleLocationLatitude", "VehicleLocationLongitude",
  "DestinationLat", "DestinationLong",
  "DistanceFromStop"
)

important_cols <- c("OriginName", "DestinationName", "NextStopPointName",
                   "VehicleLocationLatitude", "VehicleLocationLongitude", "DistanceFromStop")


for (col_name in numeric_cols) {
  if (col_name %in% names(df)) {
    df[[col_name]] <- as.numeric(df[[col_name]])
  }
}


df_clean <- df[complete.cases(df[, important_cols]), ]
print(paste("Số lượng bản ghi sau khi loại bỏ giá trị NULL:", nrow(df_clean)))




df_clean$RecordedAtTime <- ymd_hms(df_clean$RecordedAtTime)
df_clean$Hour <- hour(df_clean$RecordedAtTime)
df_clean$DayOfWeek <- wday(df_clean$RecordedAtTime)
df_clean$Month <- month(df_clean$RecordedAtTime)
df_clean$WeekOfYear <- isoweek(df_clean$RecordedAtTime)


df_clean$DayOfMonth <- day(df_clean$RecordedAtTime)
df_clean$DayOfYear <- yday(df_clean$RecordedAtTime)


df_clean$IsMonthEnd <- as.integer(day(ceiling_date(df_clean$RecordedAtTime, "month") - days(1)) == df_clean$DayOfMonth)
df_clean$IsMonthStart <- as.integer(df_clean$DayOfMonth == 1)


df_clean$IsMorningPeak <- as.integer(df_clean$Hour >= 7 & df_clean$Hour <= 9)
df_clean$IsEveningPeak <- as.integer(df_clean$Hour >= 16 & df_clean$Hour <= 18)
df_clean$IsWeekend <- as.integer(df_clean$DayOfWeek %in% c(1, 7))


df_clean$Hour_sin <- sin(2 * pi * df_clean$Hour / 24)
df_clean$Hour_cos <- cos(2 * pi * df_clean$Hour / 24)
df_clean$DayOfWeek_sin <- sin(2 * pi * df_clean$DayOfWeek / 7)
df_clean$DayOfWeek_cos <- cos(2 * pi * df_clean$DayOfWeek / 7)
df_clean$Month_sin <- sin(2 * pi * df_clean$Month / 12)
df_clean$Month_cos <- cos(2 * pi * df_clean$Month / 12)


numeric_features <- c(
  "OriginLat", "OriginLong",
  "DestinationLat", "DestinationLong",
  "VehicleLocationLatitude", "VehicleLocationLongitude",
  "Hour", "DayOfWeek", "Month", "WeekOfYear",
  "IsMorningPeak", "IsEveningPeak", "IsWeekend",
  "DayOfMonth", "DayOfYear", "IsMonthEnd", "IsMonthStart",
  "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
  "Month_sin", "Month_cos"
)

categorical_cols <- c(
  "PublishedLineName",
  "OriginName",
  "DestinationName",
  "NextStopPointName",
  "DirectionRef"
)


available_features <- intersect(numeric_features, names(df_clean))
print(paste("Sử dụng các đặc trưng:", paste(available_features, collapse = ", ")))


label_col <- "DistanceFromStop"
X <- df_clean[, available_features]
y <- df_clean[[label_col]]


for (col in available_features) {
  mean_val <- mean(X[[col]], na.rm = TRUE)
  X[[col]][is.na(X[[col]])] <- mean_val
}


scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)



df_clean <- df_clean[order(df_clean$RecordedAtTime), ]

train_size <- floor(0.7 * nrow(df_clean))
test_size <- floor(0.2 * nrow(df_clean))

train_indices <- 1:train_size
test_indices <- (train_size + 1):(train_size + test_size)
val_indices <- (train_size + test_size + 1):nrow(df_clean)

X_train <- X_scaled[train_indices, ]
y_train <- y[train_indices]
X_test <- X_scaled[test_indices, ]
y_test <- y[test_indices]
X_val <- X_scaled[val_indices, ]
y_val <- y[val_indices]

print(paste("Số lượng mẫu training:", nrow(X_train)))
print(paste("Số lượng mẫu test:", nrow(X_test)))
print(paste("Số lượng mẫu validation:", nrow(X_val)))




calculate_metrics <- function(y_true, y_pred) {
  rmse <- rmse(y_true, y_pred)
  r2 <- cor(y_true, y_pred)^2


  bins <- seq(min(y_true), max(y_true), length.out = 5)
  y_true_binned <- cut(y_true, breaks = bins, labels = FALSE, include.lowest = TRUE)
  y_pred_binned <- cut(y_pred, breaks = bins, labels = FALSE, include.lowest = TRUE)

  cm <- confusionMatrix(factor(y_pred_binned, levels=1:4), factor(y_true_binned, levels=1:4))
  return(list(rmse = rmse, r2 = r2, cm = cm, bins = bins))
}


models <- list(
  "LinearRegression" = lm,
  "Ridge" = function(X, y) {



    model <- lm(y ~ ., data = data.frame(y = y, X = X))
    return(model)
  },
  "Lasso" = function(X, y) {



    model <- lm(y ~ ., data = data.frame(y = y, X = X))
    return(model)
  },
  "RandomForest_50" = function(X, y) randomForest(x = X, y = y, ntree = 50, maxnodes = 10),
  "RandomForest_100" = function(X, y) randomForest(x = X, y = y, ntree = 100, maxnodes = 15),
  "RandomForest_200" = function(X, y) randomForest(x = X, y = y, ntree = 200, maxnodes = 15),
  "TimeSeries_RF" = function(X, y) randomForest(x = X, y = y, ntree = 100, maxnodes = 15, mtry = sqrt(ncol(X)), nodesize = 4)
)


results <- list()
all_models <- list()
confusion_matrices <- list()

print("Huấn luyện và đánh giá các mô hình...")
for (name in names(models)) {
  start_time <- Sys.time()
  try({
    print(paste("Đang huấn luyện mô hình", name, "..."))


    model <- models[[name]](X_train, y_train)
    all_models[[name]] <- model


    train_predictions <- predict(model, newdata = X_train)
    test_predictions <- predict(model, newdata = X_test)
    val_predictions <- predict(model, newdata = X_val)


    train_metrics <- calculate_metrics(y_train, train_predictions)
    test_metrics <- calculate_metrics(y_test, test_predictions)
    val_metrics <- calculate_metrics(y_val, val_predictions)


    training_time <- Sys.time() - start_time


    results[[name]] <- list(
      train_rmse = train_metrics$rmse,
      train_r2 = train_metrics$r2,
      test_rmse = test_metrics$rmse,
      test_r2 = test_metrics$r2,
      val_rmse = val_metrics$rmse,
      val_r2 = val_metrics$r2,
      training_time = as.numeric(training_time, units = "secs"),
      predictions = list(
        train = train_predictions,
        test = test_predictions,
        val = val_predictions
      )
    )

    confusion_matrices[[name]] <- list(test_cm = test_metrics$cm, cm_bins = test_metrics$bins)

    print(paste("  Train RMSE:", round(train_metrics$rmse, 4), ", R²:", round(train_metrics$r2, 4)))
    print(paste("  Test RMSE:", round(test_metrics$rmse, 4), ", R²:", round(test_metrics$r2, 4)))
    print(paste("  Validation RMSE:", round(val_metrics$rmse, 4), ", R²:", round(val_metrics$r2, 4)))
    print(paste("  Thời gian huấn luyện:", round(as.numeric(training_time, units = "secs"), 2), "giây"))

  }, error = function(e) {
    print(paste("Lỗi khi huấn luyện mô hình", name, ":", e))
  })
}


try({
  print("Đang huấn luyện mô hình XGBoost...")
  start_time <- Sys.time()


  dtrain <- xgb.DMatrix(data = X_train, label = y_train)
  dtest <- xgb.DMatrix(data = X_test, label = y_test)
  dval <- xgb.DMatrix(data = X_val, label = y_val)


  xgb_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,



    nthread = 4
  )


  xgb_model <- xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, validation = dval),
    early_stopping_rounds = 20,
    verbose = TRUE
  )
  all_models[["XGBoost"]] <- xgb_model


  xgb_train_preds <- predict(xgb_model, dtrain)
  xgb_test_preds <- predict(xgb_model, dtest)
  xgb_val_preds <- predict(xgb_model, dval)


  xgb_train_metrics <- calculate_metrics(y_train, xgb_train_preds)
  xgb_test_metrics <- calculate_metrics(y_test, xgb_test_preds)
  xgb_val_metrics <- calculate_metrics(y_val, xgb_val_preds)

  training_time <- Sys.time() - start_time


  results[["XGBoost"]] <- list(
    train_rmse = xgb_train_metrics$rmse,
    train_r2 = xgb_train_metrics$r2,
    test_rmse = xgb_test_metrics$rmse,
    test_r2 = xgb_test_metrics$r2,
    val_rmse = xgb_val_metrics$rmse,
    val_r2 = xgb_val_metrics$r2,
    training_time = as.numeric(training_time, units = "secs"),
    predictions = list(
      train = xgb_train_preds,
      test = xgb_test_preds,
      val = xgb_val_preds
    )
  )

  confusion_matrices[["XGBoost"]] <- list(test_cm = xgb_test_metrics$cm, cm_bins = xgb_test_metrics$bins)

  print(paste("  XGBoost Train RMSE:", round(xgb_train_metrics$rmse, 4), ", R²:", round(xgb_train_metrics$r2, 4)))
  print(paste("  XGBoost Test RMSE:", round(xgb_test_metrics$rmse, 4), ", R²:", round(xgb_test_metrics$r2, 4)))
  print(paste("  XGBoost Validation RMSE:", round(xgb_val_metrics$rmse, 4), ", R²:", round(xgb_val_metrics$r2, 4)))
  print(paste("  XGBoost Thời gian huấn luyện:", round(as.numeric(training_time, units = "secs"), 2), "giây"))

}, error = function(e) {
  print(paste("Lỗi khi huấn luyện mô hình XGBoost:", e))
})




if (length(results) >= 3) {
  try({
    print("Đang tạo ensemble của các mô hình tốt nhất...")


    sorted_models <- sort(sapply(results, function(x) x$test_rmse))
    top_3_models <- names(sorted_models)[1:3]

    print(paste("Top 3 models cho ensemble:", paste(top_3_models, collapse = ", ")))


    ensemble_train_preds <- rep(0, length(y_train))
    ensemble_test_preds <- rep(0, length(y_test))
    ensemble_val_preds <- rep(0, length(y_val))

    for (model_name in top_3_models) {
      ensemble_train_preds <- ensemble_train_preds + results[[model_name]]$predictions$train
      ensemble_test_preds <- ensemble_test_preds + results[[model_name]]$predictions$test
      ensemble_val_preds <- ensemble_val_preds + results[[model_name]]$predictions$val
    }


    ensemble_train_preds <- ensemble_train_preds / length(top_3_models)
    ensemble_test_preds <- ensemble_test_preds / length(top_3_models)
    ensemble_val_preds <- ensemble_val_preds / length(top_3_models)


    ensemble_train_metrics <- calculate_metrics(y_train, ensemble_train_preds)
    ensemble_test_metrics <- calculate_metrics(y_test, ensemble_test_preds)
    ensemble_val_metrics <- calculate_metrics(y_val, ensemble_val_preds)


    results[["Ensemble_Top3"]] <- list(
      train_rmse = ensemble_train_metrics$rmse,
      train_r2 = ensemble_train_metrics$r2,
      test_rmse = ensemble_test_metrics$rmse,
      test_r2 = ensemble_test_metrics$r2,
      val_rmse = ensemble_val_metrics$rmse,
      val_r2 = ensemble_val_metrics$r2,
      training_time = sum(sapply(top_3_models, function(name) results[[name]]$training_time)),
      component_models = top_3_models
    )
    confusion_matrices[["Ensemble_Top3"]] <- list(test_cm = ensemble_test_metrics$cm, cm_bins = ensemble_test_metrics$bins)

    print(paste("  Ensemble Train RMSE:", round(ensemble_train_metrics$rmse, 4), ", R²:", round(ensemble_train_metrics$r2, 4)))
    print(paste("  Ensemble Test RMSE:", round(ensemble_test_metrics$rmse, 4), ", R²:", round(ensemble_test_metrics$r2, 4)))
    print(paste("  Ensemble Validation RMSE:", round(ensemble_val_metrics$rmse, 4), ", R²:", round(ensemble_val_metrics$r2, 4)))

  }, error = function(e) {
    print(paste("Lỗi khi tạo ensemble:", e))
  })
}


if (length(results) > 0) {

  best_model_name <- names(which.min(sapply(results, function(x) x$test_rmse)))

  print(paste("Mô hình tốt nhất:", best_model_name))
  print(paste("Test RMSE:", round(results[[best_model_name]]$test_rmse, 4)))
  print(paste("Test R²:", round(results[[best_model_name]]$test_r2, 4)))
  print(paste("Thời gian huấn luyện:", round(results[[best_model_name]]$training_time, 2), "giây"))


  model_dir <- "/dbfs/FileStore/new/models"
  dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

  if (best_model_name == "XGBoost") {
    try({
      xgb.save(all_models[[best_model_name]], file.path(model_dir, "xgboost_model.model"))
      print(paste("Đã lưu mô hình XGBoost vào", file.path(model_dir, "xgboost_model.model")))
    }, error = function(e) {
      print(paste("Không thể lưu mô hình XGBoost:", e))
    })
  } else if (best_model_name != "Ensemble_Top3") {
    try({
      saveRDS(all_models[[best_model_name]], file.path(model_dir, paste0(best_model_name, "_model.rds")))
      print(paste("Đã lưu mô hình", best_model_name, "vào", file.path(model_dir, paste0(best_model_name, "_model.rds"))))
    }, error = function(e) {
      print(paste("Không thể lưu mô hình", best_model_name, ":", e))
    })
  }


  try({
    saveRDS(scaler, file.path(model_dir, "scaler.rds"))
    print(paste("Đã lưu scaler vào", file.path(model_dir, "scaler.rds")))
  }, error = function(e) {
    print(paste("Không thể lưu scaler:", e))
  })


  print("Tổng quan về kết quả các mô hình:")
  cat(sprintf("%-20s %-15s %-15s %-15s %-10s %-10s\n", "Model", "Train RMSE", "Test RMSE", "Val RMSE", "R²", "Time(s)"))
  cat(rep("-", 80), "\n")
  for (name in names(results)) {
    result <- results[[name]]
    cat(sprintf("%-20s %-15.4f %-15.4f %-15.4f %-10.4f %-10.2f\n",
                name, result$train_rmse, result$test_rmse, result$val_rmse, result$test_r2, result$training_time))
  }


  viz_dir <- "/dbfs/FileStore/new/visualizations"
  dir.create(viz_dir, recursive = TRUE, showWarnings = FALSE)


  rmse_data <- data.frame(
    Model = rep(names(results), 3),
    RMSE = c(sapply(results, function(x) x$train_rmse),
             sapply(results, function(x) x$test_rmse),
             sapply(results, function(x) x$val_rmse)),
    Type = factor(rep(c("Train", "Test", "Validation"), each = length(results)), levels = c("Train", "Test", "Validation"))
  )
  rmse_plot <- ggplot(rmse_data, aes(x = Model, y = RMSE, fill = Type)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("RMSE Comparison Across Models")
  print(rmse_plot)
  ggsave(file.path(viz_dir, "rmse_comparison.png"), rmse_plot, width = 14, height = 7)


  r2_data <- data.frame(
    Model = rep(names(results), 3),
    R2 = c(sapply(results, function(x) x$train_r2),
           sapply(results, function(x) x$test_r2),
           sapply(results, function(x) x$val_r2)),
    Type = factor(rep(c("Train", "Test", "Validation"), each = length(results)), levels = c("Train", "Test", "Validation"))
  )
  r2_plot <- ggplot(r2_data, aes(x = Model, y = R2, fill = Type)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("R² Comparison Across Models")
  print(r2_plot)
  ggsave(file.path(viz_dir, "r2_comparison.png"), r2_plot, width = 14, height = 7)


  time_data <- data.frame(
    Model = names(results),
    Time = sapply(results, function(x) x$training_time)
  )
  time_plot <- ggplot(time_data, aes(x = Model, y = Time)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("Training Time Comparison")
  print(time_plot)
  ggsave(file.path(viz_dir, "training_time_comparison.png"), time_plot, width = 14, height = 7)


  if ("XGBoost" %in% names(all_models)) {
    xgb_model <- all_models[["XGBoost"]]
    importance_matrix <- xgb.importance(model = xgb_model)
    xgb.ggplot.importance(importance_matrix)
        ggsave(file.path(viz_dir, "xgboost_feature_importance.png"), width = 14, height = 7)

    }

  for (name in names(confusion_matrices)){
      cm <- confusion_matrices[[name]]$test_cm$table
      bins <- confusion_matrices[[name]]$cm_bins

      cm_df <- as.data.frame(cm)

      bin_labels <- sapply(1:(length(bins)-1), function(i) paste(round(bins[i], 2), "-", round(bins[i+1], 2)))

      colnames(cm_df) <- bin_labels
      rownames(cm_df) <- bin_labels
        cm_plot <- ggplot(data = melt(cm), aes(x=Var2, y=Var1, fill=value)) +
        geom_tile(color = "white")+
        scale_fill_gradient2(low = "blue", high = "red", mid = "white",
        midpoint = 0, limit = c(min(cm),max(cm)), space = "Lab",
        name="Confusion Matrix") +
        theme_minimal()+
        theme(axis.text.x = element_text(angle = 45, vjust = 1,
        size = 12, hjust = 1))+
        coord_fixed() +
        geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
        labs(title = paste(name, "Confusion Matrix")) +
        theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        plot.title = element_text(size = 15, hjust = 0.5)
        )
        print(cm_plot)
    ggsave(file.path(viz_dir, paste0(name, "_confusion_matrix.png")), cm_plot, width = 14, height = 7)
        }

  if ("Ensemble_Top3" %in% names(results)) {
        ensemble_models <- results[["Ensemble_Top3"]]$component_models
        ensemble_results <- data.frame(
        Model = c(ensemble_models, "Ensemble_Top3"),
        Test_RMSE = c(sapply(ensemble_models, function(x) results[[x]]$test_rmse), results[["Ensemble_Top3"]]$test_rmse),
        Test_R2 = c(sapply(ensemble_models, function(x) results[[x]]$test_r2), results[["Ensemble_Top3"]]$test_r2)
    )
    ensemble_comparison_plot <- ggplot(ensemble_results, aes(x = Model)) +
    geom_col(aes(y = Test_RMSE, fill = "Test RMSE"), position = position_dodge(width=0.7), width = 0.3) +
    geom_col(aes(y = Test_R2, fill = "Test R2"), position = position_dodge(width=0.7), width = 0.3) +
    scale_fill_manual(values = c("Test RMSE" = "blue", "Test R2" = "red")) +
    labs(title = "Ensemble vs Component Models Performance") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
    print(ensemble_comparison_plot)
  ggsave(file.path(viz_dir, paste0("Ensemble_vs_models.png")), ensemble_comparison_plot, width = 14, height = 7)

    }
  print(paste("Đã lưu các hình ảnh trực quan vào", viz_dir))

} else {
  print("Không có mô hình nào được huấn luyện thành công để so sánh")
}

print("Phân tích đã hoàn thành thành công!")


if ("XGBoost" %in% names(all_models)) {
  print("Thực hiện phân tích chi tiết về các đặc trưng quan trọng cho mô hình XGBoost...")


  xgb_model <- all_models[["XGBoost"]]
  importance_matrix <- xgb.importance(model = xgb_model)
  top_10_features <- head(importance_matrix$Feature, 10)
  print(paste("Top 10 đặc trưng quan trọng nhất:", paste(top_10_features, collapse = ", ")))

  viz_dir <- "/dbfs/FileStore/new/visualizations"
  print("\nCreating feature dependence plots...")


  X_test_df = data.frame(X_test)

  for (i in 1:min(5, length(top_10_features))) {
        feature <- top_10_features[i]

        if (feature %in% colnames(X_test_df)) {
            feature_values <- X_test_df[, feature]
            predictions <- results[["XGBoost"]]$predictions$test

            plot_data <- data.frame(feature_values = feature_values, predictions = predictions)

            dependence_plot <- ggplot(plot_data, aes(x = feature_values, y = predictions)) +
                geom_point(alpha = 0.3, size = 1) +
                geom_smooth(method = "lm", color = "red", se = FALSE) +
                labs(title = paste("Impact of", feature),
                     x = feature,
                     y = "Predicted values") +
                theme_minimal()

            print(dependence_plot)
            ggsave(file.path(viz_dir, paste0("dependence_", feature, ".png")), dependence_plot, width = 6, height = 4)
            } else {
                print(paste("Feature", feature, "not found in X_test_df"))
                }
    }

    print("\nCreating feature correlation heatmap...")

    top_feature_indices <- which(colnames(X_test) %in% top_10_features)

    X_test_top <- X_test[, top_feature_indices]

    corr_matrix <- cor(X_test_top)

    heatmap_plot <- ggplot(data = melt(corr_matrix), aes(x=Var1, y=Var2, fill=value)) +
        geom_tile(color = "white") +
        scale_fill_gradient2(low= "blue", high = "red", mid = "white",
                                midpoint = 0, limit = c(-1,1), space = "Lab",
                                name="Correlation") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1,
                                           size = 12, hjust = 1)) +
        coord_fixed() +
        geom_text(aes(Var2, Var1, label = round(value, 2)), color = "black", size = 4) +
        labs(title = "Correlation between top 10 important features") +
        theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            panel.border = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            plot.title = element_text(size = 15, hjust = 0.5)
        )
    print(heatmap_plot)
    ggsave(file.path(viz_dir, "feature_correlation.png"), heatmap_plot, width = 10, height = 8)


  xgb_importance_plot <- xgb.ggplot.importance(importance_matrix) +
        labs(title = "XGBoost Feature Importance")
  print(xgb_importance_plot)
  ggsave(file.path(viz_dir, "xgboost_feature_importance_detailed.png"), xgb_importance_plot, width = 10, height = 6)

    print(paste("Đã lưu các hình ảnh phân tích đặc trưng vào", viz_dir))
} else {
  print("Không có mô hình XGBoost nào được huấn luyện để phân tích đặc trưng")
}

print("Phân tích đã hoàn thành!")
