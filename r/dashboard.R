suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(readr)
  library(dplyr)
  library(scales)
})

root_dir <- normalizePath("..", mustWork = FALSE)
if (!dir.exists(file.path(root_dir, "outputs"))) {
  root_dir <- normalizePath(".", mustWork = FALSE)
}
outputs_dir <- file.path(root_dir, "outputs")

scores_path <- file.path(outputs_dir, "model_scores_r.csv")
preds_path <- file.path(outputs_dir, "test_predictions_r.csv")
featured_path <- file.path(outputs_dir, "featured_data.csv")

if (!file.exists(scores_path) || !file.exists(preds_path) || !file.exists(featured_path)) {
  stop("Run python/data_pipeline.py and r/train_model.R first.")
}

scores <- read_csv(scores_path, show_col_types = FALSE)
preds <- read_csv(preds_path, show_col_types = FALSE)
featured <- read_csv(featured_path, show_col_types = FALSE)
preds <- preds %>% mutate(Date = as.Date(Date), actual = as.integer(actual), pred_glm = as.integer(pred_glm), pred_rf = as.integer(pred_rf))
featured <- featured %>% mutate(Date = as.Date(Date))

latest_close <- dplyr::last(featured$Close)
close_5d_ago <- dplyr::lag(featured$Close, 5) %>% dplyr::last()
recent_return <- ifelse(is.na(close_5d_ago) || close_5d_ago == 0, NA_real_, (latest_close / close_5d_ago) - 1)
best_row <- scores %>% arrange(desc(accuracy)) %>% slice(1)

ui <- fluidPage(
  titlePanel("Stock Trend Intelligence Dashboard (R Shiny)"),
  fluidRow(
    column(4, tags$div(style = "padding:12px;background:#f8f9fa;border-radius:10px;margin-bottom:10px;", tags$h4("Latest Close"), tags$h3(sprintf("$%.2f", latest_close)))),
    column(4, tags$div(style = "padding:12px;background:#f8f9fa;border-radius:10px;margin-bottom:10px;", tags$h4("5-Day Return"), tags$h3(percent(recent_return, accuracy = 0.01)))),
    column(4, tags$div(style = "padding:12px;background:#f8f9fa;border-radius:10px;margin-bottom:10px;", tags$h4("Best Model"), tags$h3(best_row$model)))
  ),
  sidebarLayout(
    sidebarPanel(
      selectInput("pred_model", "Prediction Model", choices = c("Random Forest" = "pred_rf", "Logistic Regression" = "pred_glm"), selected = "pred_rf"),
      sliderInput("lookback_days", "Price Chart Lookback (days)", min = 120, max = 2000, value = 600, step = 30)
    ),
    mainPanel(
      plotlyOutput("pricePlot"),
      plotlyOutput("scorePlot"),
      plotlyOutput("signalPlot"),
      plotlyOutput("predPlot")
    )
  )
)

server <- function(input, output, session) {
  filtered_price <- reactive({
    cutoff <- max(featured$Date, na.rm = TRUE) - input$lookback_days
    featured %>% filter(Date >= cutoff)
  })

  output$pricePlot <- renderPlotly({
    p <- filtered_price()
    plot_ly(p, x = ~Date, y = ~Close, type = "scatter", mode = "lines", name = "Close") %>%
      layout(title = "Close Price Trend", yaxis = list(title = "Price"), xaxis = list(title = "Date"))
  })

  output$scorePlot <- renderPlotly({
    plot_ly(scores, x = ~model, y = ~accuracy, type = "bar", text = ~percent(accuracy, accuracy = 0.01), textposition = "outside") %>%
      layout(title = "Model Accuracy Comparison", yaxis = list(title = "Accuracy", tickformat = ".0%"))
  })

  output$signalPlot <- renderPlotly({
    model_col <- input$pred_model
    signal_df <- preds %>%
      mutate(
        model_signal = .data[[model_col]],
        is_correct = as.integer(model_signal == actual)
      ) %>%
      group_by(Date) %>%
      summarise(hit_rate = mean(is_correct), .groups = "drop")

    plot_ly(signal_df, x = ~Date, y = ~hit_rate, type = "scatter", mode = "lines") %>%
      layout(title = "Prediction Hit Rate Over Time", yaxis = list(title = "Hit Rate", tickformat = ".0%"))
  })

  output$predPlot <- renderPlotly({
    model_col <- input$pred_model
    chart_df <- preds %>%
      mutate(predicted_label = ifelse(.data[[model_col]] == 1, "UP", "DOWN"))

    plot_ly(chart_df, x = ~predicted_label, type = "histogram") %>%
      layout(title = "Prediction Direction Distribution", xaxis = list(title = "Predicted Direction"), yaxis = list(title = "Count"))
  })
}

shinyApp(ui, server)
