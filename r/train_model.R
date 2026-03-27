suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
  library(randomForest)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
} else {
  script_dir <- getwd()
}
root_dir <- normalizePath(file.path(script_dir, ".."), mustWork = FALSE)
if (!dir.exists(file.path(root_dir, "archive"))) {
  root_dir <- script_dir
}
outputs_dir <- file.path(root_dir, "outputs")
models_dir <- file.path(root_dir, "models")
dir.create(outputs_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

featured_path <- file.path(outputs_dir, "featured_data.csv")
if (!file.exists(featured_path)) {
  stop("Run python/data_pipeline.py first to generate outputs/featured_data.csv")
}

df <- read_csv(featured_path, show_col_types = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    target_direction = as.factor(target_direction)
  ) %>%
  arrange(Date)

feature_cols <- c("return_1d", "return_5d", "ma_10", "ma_20", "volatility_10", "volume_change_1d")
split_idx <- floor(nrow(df) * 0.8)
train_df <- df[1:split_idx, ]
test_df <- df[(split_idx + 1):nrow(df), ]

baseline_pred <- factor(rep(0, nrow(test_df)), levels = levels(test_df$target_direction))
baseline_acc <- mean(baseline_pred == test_df$target_direction)

glm_formula <- as.formula(paste("target_direction ~", paste(feature_cols, collapse = " + ")))
glm_fit <- glm(glm_formula, data = train_df, family = "binomial")
glm_prob <- predict(glm_fit, newdata = test_df, type = "response")
glm_pred <- factor(ifelse(glm_prob > 0.5, 1, 0), levels = levels(test_df$target_direction))
glm_acc <- mean(glm_pred == test_df$target_direction)

rf_fit <- randomForest(glm_formula, data = train_df, ntree = 300)
rf_pred <- predict(rf_fit, newdata = test_df)
rf_acc <- mean(rf_pred == test_df$target_direction)

best_model <- if (rf_acc >= glm_acc) "random_forest" else "logistic_regression"
if (best_model == "random_forest") {
  saveRDS(rf_fit, file.path(models_dir, "best_model_r.rds"))
} else {
  saveRDS(glm_fit, file.path(models_dir, "best_model_r.rds"))
}

scores <- tibble(
  model = c("baseline_always_down", "logistic_regression", "random_forest"),
  accuracy = c(baseline_acc, glm_acc, rf_acc)
)
write_csv(scores, file.path(outputs_dir, "model_scores_r.csv"))

pred_out <- test_df %>%
  transmute(Date, actual = target_direction, pred_glm = glm_pred, pred_rf = rf_pred)
write_csv(pred_out, file.path(outputs_dir, "test_predictions_r.csv"))

cat(sprintf("R training complete. Best model: %s\n", best_model))
