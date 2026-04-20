suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
  library(randomForest)
  library(zoo)
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
raw_data_path <- file.path(root_dir, "data", "stock data.csv")

if (!file.exists(featured_path)) {
  if (file.exists(raw_data_path)) {
    message("featured_data.csv not found. Building it from data/stock data.csv ...")
    raw_df <- read_csv(raw_data_path, show_col_types = FALSE) %>%
      rename(
        Date = date,
        Open = open,
        High = high,
        Low = low,
        Close = close,
        Volume = volume
      ) %>%
      mutate(
        Date = dmy(Date),
        Name = as.character(Name)
      ) %>%
      filter(!is.na(Date), !is.na(Close), !is.na(Volume))

    ticker <- if ("AAL" %in% raw_df$Name) "AAL" else names(sort(table(raw_df$Name), decreasing = TRUE))[1]
    featured_df <- raw_df %>%
      filter(Name == ticker) %>%
      arrange(Date) %>%
      mutate(
        return_1d = Close / lag(Close, 1) - 1,
        return_5d = Close / lag(Close, 5) - 1,
        ma_10 = zoo::rollmean(Close, 10, fill = NA, align = "right"),
        ma_20 = zoo::rollmean(Close, 20, fill = NA, align = "right"),
        volatility_10 = zoo::rollapply(return_1d, 10, sd, fill = NA, align = "right"),
        volume_change_1d = Volume / lag(Volume, 1) - 1,
        future_return_1d = lead(Close, 1) / Close - 1,
        target_direction = as.integer(future_return_1d > 0)
      ) %>%
      select(
        Date, Open, High, Low, Close, Volume, return_1d, return_5d, ma_10, ma_20,
        volatility_10, volume_change_1d, future_return_1d, target_direction
      ) %>%
      filter(
        if_all(c(return_1d, return_5d, ma_10, ma_20, volatility_10, volume_change_1d, future_return_1d), ~ !is.na(.x))
      )

    write_csv(featured_df, featured_path)
  } else {
    stop("Missing both outputs/featured_data.csv and data/stock data.csv")
  }
}

df <- read_csv(featured_path, show_col_types = FALSE) %>%
  mutate(
    Date = as.Date(Date),
    target_direction = as.factor(target_direction)
  ) %>%
  arrange(Date)

feature_cols <- c("return_1d", "return_5d", "ma_10", "ma_20", "volatility_10", "volume_change_1d")
missing_features <- setdiff(feature_cols, colnames(df))
if (length(missing_features) > 0) {
  stop(sprintf("Missing required feature columns: %s", paste(missing_features, collapse = ", ")))
}

split_idx <- floor(nrow(df) * 0.8)
train_df <- df[1:split_idx, ]
test_df <- df[(split_idx + 1):nrow(df), ]

baseline_pred <- factor(rep(0, nrow(test_df)), levels = levels(test_df$target_direction))
baseline_acc <- mean(baseline_pred == test_df$target_direction)
baseline_f1 <- 0

glm_formula <- as.formula(paste("target_direction ~", paste(feature_cols, collapse = " + ")))
glm_fit <- glm(glm_formula, data = train_df, family = "binomial")
glm_prob <- predict(glm_fit, newdata = test_df, type = "response")
glm_pred <- factor(ifelse(glm_prob > 0.5, 1, 0), levels = levels(test_df$target_direction))
glm_acc <- mean(glm_pred == test_df$target_direction)

rf_fit <- randomForest(glm_formula, data = train_df, ntree = 400, mtry = 3, importance = TRUE)
rf_pred <- predict(rf_fit, newdata = test_df)
rf_acc <- mean(rf_pred == test_df$target_direction)

f1_score <- function(actual, pred) {
  actual_i <- as.integer(as.character(actual))
  pred_i <- as.integer(as.character(pred))
  tp <- sum(actual_i == 1 & pred_i == 1)
  fp <- sum(actual_i == 0 & pred_i == 1)
  fn <- sum(actual_i == 1 & pred_i == 0)
  if ((2 * tp + fp + fn) == 0) return(0)
  (2 * tp) / (2 * tp + fp + fn)
}

glm_f1 <- f1_score(test_df$target_direction, glm_pred)
rf_f1 <- f1_score(test_df$target_direction, rf_pred)

best_model <- if (rf_acc >= glm_acc) "random_forest" else "logistic_regression"
if (best_model == "random_forest") {
  saveRDS(rf_fit, file.path(models_dir, "best_model_r.rds"))
} else {
  saveRDS(glm_fit, file.path(models_dir, "best_model_r.rds"))
}

scores <- tibble(
  model = c("baseline_always_down", "logistic_regression", "random_forest"),
  accuracy = c(baseline_acc, glm_acc, rf_acc),
  f1_up_class = c(baseline_f1, glm_f1, rf_f1)
)
write_csv(scores, file.path(outputs_dir, "model_scores_r.csv"))

pred_out <- test_df %>%
  transmute(Date, actual = target_direction, pred_glm = glm_pred, pred_rf = rf_pred)
write_csv(pred_out, file.path(outputs_dir, "test_predictions_r.csv"))

importance_tbl <- tibble(
  feature = rownames(importance(rf_fit)),
  gini_importance = as.numeric(importance(rf_fit)[, "MeanDecreaseGini"])
) %>%
  arrange(desc(gini_importance))
write_csv(importance_tbl, file.path(outputs_dir, "feature_importance_r.csv"))

cat(sprintf("R training complete. Best model: %s (Accuracy %.4f)\n", best_model, max(glm_acc, rf_acc)))
