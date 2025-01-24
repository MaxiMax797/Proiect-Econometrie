required_packages <- c(
  "tidyverse", "readxl", "caret", "rpart", "randomForest", "e1071", 
  "xgboost", "neuralnet", "glmnet", "ggplot2", "writexl", "gridExtra", 
  "grDevices", "dplyr", "zoo", "car", "nortest", "Boruta", "lmtest"
)

install_if_missing <- function(packages) {
  for(package in packages) {
    if(!require(package, character.only = TRUE, quietly = TRUE)) {
      install.packages(package)
      library(package, character.only = TRUE)
    }
  }
}

install_if_missing(required_packages)

library(tidyverse)
library(readxl)
library(caret)
library(rpart)
library(randomForest)
library(e1071)
library(xgboost)
library(neuralnet)
library(glmnet)
library(ggplot2)
library(writexl)
library(gridExtra)
library(grDevices)
library(dplyr)
library(car)
library(nortest)
library(Boruta)

# Citirea datelor
df <- read_excel("frigidere_emag_final.xlsx", sheet = "Frigidere")

df <- df %>%
  mutate(
    volum_inaltime_ratio = Volum / Inaltime,
    volum_patrat = Volum^2,
    inaltime_patrat = Inaltime^2,
    volum_per_usa = case_when(
      O_usa == 1 ~ Volum,
      Doua_usi == 1 ~ Volum/2,
      TRUE ~ Volum
    )
  )

features <- c('Volum', 'Inaltime', 'Alb', 'Argintiu', 'O_usa', 'Doua_usi', 'Minibar',
              'consum_energetic', 'volum_inaltime_ratio', 'volum_patrat', 
              'inaltime_patrat', 'volum_per_usa')

# Maparea claselor energetice
clasa_mapping <- c('A+++' = 7, 'A++' = 6, 'A+' = 5, 'A' = 4, 'B' = 3, 'C' = 2, 'D' = 1, 'E' = 0)
df$clasa_numeric <- clasa_mapping[df$clasa]
features <- c(features, 'clasa_numeric')

# Pregatirea datelor
X <- df[features] %>% replace(is.na(.), 0)
y <- df$pret

# Identificarea si tratarea outlierelor folosind IQR
Q1 <- quantile(y, 0.25)
Q3 <- quantile(y, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Filtrarea outlierilor extremi
valid_indices <- which(y >= lower_bound & y <= upper_bound)
X <- X[valid_indices,]
y <- y[valid_indices]

# Transformare logaritmica pentru pret
y_log <- log(y)

# Impartirea datelor in training si test (80-20)
set.seed(42)
train_index <- createDataPartition(y_log, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y_log[train_index]
y_test <- y_log[-train_index]

# Scalarea datelor
preproc <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preproc, X_train)
X_test_scaled <- predict(preproc, X_test)

# Control pentru cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  verboseIter = TRUE
)

# Functie pentru calculul metricilor de performanta adaptata pentru transformarea logaritmica
calculate_metrics <- function(actual, predicted) {
  actual_orig <- exp(actual)
  predicted_orig <- exp(predicted)
  
  mae <- mean(abs(actual_orig - predicted_orig))
  mse <- mean((actual_orig - predicted_orig)^2)
  rmse <- sqrt(mse)
  mape <- mean(abs((actual_orig - predicted_orig) / actual_orig)) * 100
  r2 <- 1 - sum((actual_orig - predicted_orig)^2) / sum((actual_orig - mean(actual_orig))^2)
  
  c(
    MAE = mae,
    MSE = mse,
    RMSE = rmse,
    MAPE = mape,
    R2 = r2
  )
}

# Lista pentru stocarea modelelor și performantelor
models <- list()
model_performances <- list()

# Regresie Simpla
print("Training Simple Regression...")
simple_reg_model <- lm(y_train ~ Volum, data = as.data.frame(X_train_scaled))
models[["Simple Regression"]] <- simple_reg_model
simple_reg_pred <- predict(simple_reg_model, as.data.frame(X_test_scaled))
model_performances[["Simple Regression"]] <- calculate_metrics(y_test, simple_reg_pred)

# Testarea ipotezelor pentru regresie liniară
print("Testing Linear Regression Assumptions...")
# Testul Breusch-Pagan pentru homoscedacticitate
bptest(simple_reg_model)
# Testul Durbin-Watson pentru autocorelare
durbinWatsonTest(simple_reg_model)
# Testul Shapiro-Wilk pentru normalitatea reziduurilor
shapiro.test(residuals(simple_reg_model))

# Regresie Multipla
print("Training Multiple Regression...")
multiple_reg_model <- lm(y_train ~ ., data = as.data.frame(X_train_scaled))
models[["Multiple Regression"]] <- multiple_reg_model
multiple_reg_pred <- predict(multiple_reg_model, as.data.frame(X_test_scaled))
model_performances[["Multiple Regression"]] <- calculate_metrics(y_test, multiple_reg_pred)

# Testul F pentru semnificația statistică a regresiei multiple
print("Performing F-test for Multiple Regression...")
summary(multiple_reg_model)$fstatistic

# Random Forest
print("Training Random Forest...")
rf_grid <- expand.grid(
  mtry = seq(floor(sqrt(ncol(X_train_scaled))), ncol(X_train_scaled), length.out = 5)
)
rf_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 1000
)
models[["Random Forest"]] <- rf_model
rf_pred <- predict(rf_model, X_test_scaled)
model_performances[["Random Forest"]] <- calculate_metrics(y_test, rf_pred)

# XGBoost
print("Training XGBoost...")
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)
xgb_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid
)
models[["XGBoost"]] <- xgb_model
xgb_pred <- predict(xgb_model, X_test_scaled)
model_performances[["XGBoost"]] <- calculate_metrics(y_test, xgb_pred)

# Neural Network
print("Training Neural Network...")
nnet_grid <- expand.grid(
  size = c(5, 10, 15),
  decay = c(0.001, 0.01, 0.1)
)
nnet_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = nnet_grid,
  linout = TRUE,
  trace = FALSE,
  maxit = 1000
)
models[["Neural Network"]] <- nnet_model
nnet_pred <- predict(nnet_model, X_test_scaled)
model_performances[["Neural Network"]] <- calculate_metrics(y_test, nnet_pred)

# Rețea Neuronală Artificială (ANN)
print("Training Artificial Neural Network...")
ann_model <- neuralnet(y_train ~ ., data = as.data.frame(X_train_scaled), 
                       hidden = c(8, 4), act.fct = "logistic", linear.output = TRUE)
ann_pred <- compute(ann_model, as.data.frame(X_test_scaled))$net.result
model_performances[["ANN"]] <- calculate_metrics(y_test, ann_pred)
models[["ANN"]] <- ann_model

# SVM
print("Training SVM...")
svm_grid <- expand.grid(
  sigma = seq(0.01, 0.1, length = 5),
  C = seq(0.5, 2, length = 5)
)
svm_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "svmRadial",
  trControl = ctrl,
  tuneGrid = svm_grid
)
models[["SVM"]] <- svm_model
svm_pred <- predict(svm_model, X_test_scaled)
model_performances[["SVM"]] <- calculate_metrics(y_test, svm_pred)

# Lasso
print("Training Lasso...")
lasso_grid <- expand.grid(
  alpha = 1,
  lambda = 10^seq(-5, 1, length = 100)
)
lasso_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = lasso_grid,
  preProcess = c("center", "scale")
)
models[["Lasso"]] <- lasso_model
lasso_pred <- predict(lasso_model, X_test_scaled)
model_performances[["Lasso"]] <- calculate_metrics(y_test, lasso_pred)

# Elastic Net
print("Training Elastic Net...")
elastic_grid <- expand.grid(
  alpha = seq(0, 1, length = 10),
  lambda = 10^seq(-5, 1, length = 100)
)
elastic_model <- train(
  x = X_train_scaled,
  y = y_train,
  method = "glmnet",
  trControl = ctrl,
  tuneGrid = elastic_grid,
  preProcess = c("center", "scale")
)
models[["Elastic Net"]] <- elastic_model
elastic_pred <- predict(elastic_model, X_test_scaled)
model_performances[["Elastic Net"]] <- calculate_metrics(y_test, elastic_pred)

# Analiza Boruta
print("Performing Boruta Feature Selection...")
boruta_model <- Boruta(y_train ~ ., data = as.data.frame(X_train_scaled), doTrace = 2)
important_features <- getSelectedAttributes(boruta_model, withTentative = FALSE)

# Cele mai importante caracteristici
print("Top Important Features:")
print(important_features)

print("Confirmed Attributes:")
print(boruta_model$finalDecision)

# Prepare Boruta feature selection results
boruta_features_df <- data.frame( 
  Feature = important_features, 
  Importance_Status = boruta_model$finalDecision[important_features] 
)

# Convertim performantele în dataframe
performance_df <- do.call(rbind, model_performances) %>%
  as.data.frame() %>%
  mutate(Model = rownames(.)) %>%
  select(Model, everything())

# Pregatirea datelor pentru clustering
model_metrics <- performance_df %>%
  select(MAE, RMSE, MAPE, R2) %>%
  scale()

# Aplicare k-means
set.seed(42)
k <- 3
clusters <- kmeans(model_metrics, centers = k)

# Adaugare cluster la dataframe
performance_df$Cluster <- as.factor(clusters$cluster)

# Selectarea celor mai bune modele din clusterul cu performanta cea mai buna
cluster_performance <- aggregate(R2 ~ Cluster, data = performance_df, FUN = mean)
best_cluster <- cluster_performance$Cluster[which.max(cluster_performance$R2)]

best_models <- performance_df %>%
  filter(Cluster == best_cluster) %>%
  arrange(desc(R2)) %>%
  head(2)

print("Cele mai bune doua modele sunt:")
print(best_models)

# Cream dataframe cu predictii pentru cele mai bune modele
predictions_df <- data.frame(
  Product_Index = seq_along(y_test),
  Actual_Price = exp(y_test)
)

# Adaugam predictiile pentru cele mai bune doua modele
for(model_name in best_models$Model) {
  if(model_name == "ANN") {
    predictions <- exp(compute(models[[model_name]], as.data.frame(X_test_scaled))$net.result)
  } else {
    predictions <- exp(predict(models[[model_name]], X_test_scaled))
  }
  predictions_df[[paste0(model_name, "_Predicted")]] <- predictions
}

# Functia pentru ploturi
create_enhanced_line_plots <- function(predictions_df, best_models) {
  # Color palette
  model_colors <- c("#2E86C1", "#E67E22", "#2ECC71")
  
  # Plot 1: Enhanced comparison plot with markers
  comparison_plot_enhanced <- ggplot() +
    geom_line(data = predictions_df, 
              aes(x = Product_Index, y = Actual_Price),
              color = "#34495E",
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index, y = Actual_Price),
               color = "#34495E",
               size = 2) +
    geom_line(data = predictions_df,
              aes(x = Product_Index, 
                  y = .data[[paste0(best_models$Model[1], "_Predicted")]],
                  color = best_models$Model[1]),
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index,
                   y = .data[[paste0(best_models$Model[1], "_Predicted")]],
                   color = best_models$Model[1]),
               size = 2) +
    geom_line(data = predictions_df,
              aes(x = Product_Index,
                  y = .data[[paste0(best_models$Model[2], "_Predicted")]],
                  color = best_models$Model[2]),
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index,
                   y = .data[[paste0(best_models$Model[2], "_Predicted")]],
                   color = best_models$Model[2]),
               size = 2) +
    scale_color_manual(values = model_colors) + theme_minimal() +
    labs(title = "Comparison of Actual vs Predicted Prices",
         subtitle = paste("Best performing models:", 
                          paste(best_models$Model, collapse = " and ")),
         x = "Product Index",
         y = "Price (RON)",
         color = "Model Type") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray95")
    )
  
  # Plot 2: Stacked comparison with markers
  stacked_comparison <- ggplot() +
    geom_line(data = predictions_df,
              aes(x = Product_Index, y = Actual_Price + 2000),
              color = "#34495E",
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index, y = Actual_Price + 2000),
               color = "#34495E",
               size = 2) +
    geom_line(data = predictions_df,
              aes(x = Product_Index,
                  y = .data[[paste0(best_models$Model[1], "_Predicted")]]),
              color = model_colors[1],
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index,
                   y = .data[[paste0(best_models$Model[1],"_Predicted")]]),
               color = model_colors[1],
               size = 2) +
    geom_line(data = predictions_df,
              aes(x = Product_Index,
                  y = .data[[paste0(best_models$Model[2], "_Predicted")]] - 2000),
              color = model_colors[2],
              size = 1) +
    geom_point(data = predictions_df,
               aes(x = Product_Index,
                   y = .data[[paste0(best_models$Model[2], "_Predicted")]] - 2000),
               color = model_colors[2],
               size = 2) +
    theme_minimal() +
    labs(title = "Stacked View of Price Predictions",
         subtitle = "Models shown with vertical offset for clarity",
         x = "Product Index",
         y = "Price (RON)",
         color = "Model Type") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray95")
    )
  
  # Plot 3: Percentage difference from actual
  diff_df <- predictions_df %>%
    mutate(
      Diff_Model1 = (.data[[paste0(best_models$Model[1], "_Predicted")]] - Actual_Price) / Actual_Price * 100,
      Diff_Model2 = (.data[[paste0(best_models$Model[2], "_Predicted")]] - Actual_Price) / Actual_Price * 100
    )
  
  difference_plot <- ggplot(diff_df) +
    geom_line(aes(x = Product_Index, y = Diff_Model1, color = best_models$Model[1]), size = 1) +
    geom_point(aes(x = Product_Index, y = Diff_Model1, color = best_models$Model[1]), size = 2) +
    geom_line(aes(x = Product_Index, y = Diff_Model2, color = best_models$Model[2]), size = 1) +
    geom_point(aes(x = Product_Index, y = Diff_Model2, color = best_models$Model[2]), size = 2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = model_colors[1:2]) +
    theme_minimal() +
    labs(title = "Percentage Difference from Actual Prices",
         subtitle = "Model prediction deviation from actual values",
         x = "Product Index",
         y = "Difference (%)",
         color = "Model") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray95")
    )
  
  #Returneaza lista
  list(
    comparison = comparison_plot_enhanced,
    stacked = stacked_comparison,
    difference = difference_plot
  )
}

enhanced_plots <- create_enhanced_line_plots(predictions_df, best_models)

ggsave("enhanced_comparison_plot.png", enhanced_plots$comparison, width = 15, height = 8, dpi = 300)
ggsave("enhanced_stacked_plot.png", enhanced_plots$stacked, width = 15, height = 8, dpi = 300)
ggsave("enhanced_difference_plot.png", enhanced_plots$difference, width = 15, height = 8, dpi = 300)

# Predict for entire original dataset
X_full <- df[features] %>% replace(is.na(.), 0)
X_full_scaled <- predict(preproc, X_full)

# Predict using best models
predictions_full_df <- data.frame(
  Product_Index = seq_len(nrow(X_full)),
  Original_Price = df$pret
)

# Add predictions from top models
for(model_name in best_models$Model) {
  if(model_name == "ANN") {
    full_predictions <- round(exp(compute(models[[model_name]], as.data.frame(X_full_scaled))$net.result), 2)
  } else {
    full_predictions <- round(exp(predict(models[[model_name]], X_full_scaled)), 2)
  }
  predictions_full_df[[paste0(model_name, "_Predicted")]] <- full_predictions
  
  # Calculate prediction error metrics
  predictions_full_df[[paste0(model_name, "_AbsoluteError")]] <- 
    round(abs(predictions_full_df$Original_Price - full_predictions), 2)
  predictions_full_df[[paste0(model_name, "_PercentError")]] <- 
    round(abs(predictions_full_df$Original_Price - full_predictions) / 
            predictions_full_df$Original_Price * 100, 2)
}

# Write results to multiple formats
write_xlsx(list(
  "Performanta Modele" = performance_df %>% arrange(desc(R2)),
  "Cele mai bune modele" = best_models,
  "Predictii Toate Frigiderele" = predictions_full_df,
  "Selectie Caracteristici Boruta" = boruta_features_df
), path = "rezultate_predictii_complete.xlsx")

write.csv(predictions_full_df, "predictii_frigidere_complete.csv", row.names = FALSE)
write.csv(performance_df %>% arrange(desc(R2)), "performanta_modele.csv", row.names = FALSE)
write.csv(best_models, "cele_mai_bune_modele.csv", row.names = FALSE)
write.csv(boruta_features_df, "selectie_caracteristici_boruta.csv", row.names = FALSE)
print("Predictions generated for all refrigerators. Results saved in Excel and CSV formats.")