suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(readr)
  library(dplyr)
  library(scales)
  library(bslib)
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
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#3B82F6",
    base_font = font_google("Inter")
  ),
  tags$head(
    tags$style(HTML("
      .app-title {
        margin-bottom: 4px;
        font-weight: 700;
        letter-spacing: 0.2px;
      }
      .app-subtitle {
        color: #64748B;
        margin-bottom: 14px;
      }
      .kpi-card {
        padding: 14px;
        background: linear-gradient(145deg, #F8FAFC, #EEF2FF);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        margin-bottom: 12px;
      }
      .kpi-label {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 6px;
      }
      .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
      }
      .sidebar-panel {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px;
      }
    "))
  ),
  div(
    class = "app-title h2",
    "Stock Trend Intelligence Dashboard (R Shiny)"
  ),
  div(
    class = "app-subtitle",
    "Elegant analytics view of price action, model quality, and prediction performance."
  ),
  fluidRow(
    column(
      4,
      div(class = "kpi-card",
          div(class = "kpi-label", "Latest Close"),
          div(class = "kpi-value", sprintf("$%.2f", latest_close)))
    ),
    column(
      4,
      div(class = "kpi-card",
          div(class = "kpi-label", "5-Day Return"),
          div(class = "kpi-value", percent(recent_return, accuracy = 0.01)))
    ),
    column(
      4,
      div(class = "kpi-card",
          div(class = "kpi-label", "Best Model"),
          div(class = "kpi-value", best_row$model))
    )
  ),
  sidebarLayout(
    sidebarPanel(
      class = "sidebar-panel",
      selectInput("pred_model", "Prediction Model", choices = c("Random Forest" = "pred_rf", "Logistic Regression" = "pred_glm"), selected = "pred_rf"),
      sliderInput("lookback_days", "Price Chart Lookback (days)", min = 120, max = 2000, value = 600, step = 30),
      hr(),
      helpText("Tip: use a shorter lookback for cleaner trend inspection, and switch models to compare signal behavior.")
    ),
    mainPanel(
      fluidRow(
        column(12, plotlyOutput("pricePlot", height = "320px"))
      ),
      fluidRow(
        column(6, plotlyOutput("scorePlot", height = "300px")),
        column(6, plotlyOutput("predPlot", height = "300px"))
      ),
      fluidRow(
        column(12, plotlyOutput("signalPlot", height = "320px"))
      )
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
    plot_ly(
      p,
      x = ~Date, y = ~Close,
      type = "scatter", mode = "lines",
      name = "Close",
      line = list(color = "#2563EB", width = 2.2)
    ) %>%
      layout(
        title = "Close Price Trend",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        yaxis = list(title = "Price", gridcolor = "#E2E8F0"),
        xaxis = list(title = "Date", gridcolor = "#F1F5F9")
      )
  })

  output$scorePlot <- renderPlotly({
    plot_ly(
      scores,
      x = ~model, y = ~accuracy,
      type = "bar",
      marker = list(color = c("#94A3B8", "#38BDF8", "#2563EB")),
      text = ~percent(accuracy, accuracy = 0.01),
      textposition = "outside"
    ) %>%
      layout(
        title = "Model Accuracy Comparison",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        yaxis = list(title = "Accuracy", tickformat = ".0%", gridcolor = "#E2E8F0"),
        xaxis = list(title = "")
      )
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

    plot_ly(
      signal_df,
      x = ~Date, y = ~hit_rate,
      type = "scatter", mode = "lines",
      line = list(color = "#0EA5E9", width = 2.2),
      fill = "tozeroy",
      fillcolor = "rgba(14,165,233,0.12)"
    ) %>%
      layout(
        title = "Prediction Hit Rate Over Time",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        yaxis = list(title = "Hit Rate", tickformat = ".0%", gridcolor = "#E2E8F0"),
        xaxis = list(title = "Date", gridcolor = "#F1F5F9")
      )
  })

  output$predPlot <- renderPlotly({
    model_col <- input$pred_model
    chart_df <- preds %>%
      mutate(predicted_label = ifelse(.data[[model_col]] == 1, "UP", "DOWN"))

    plot_ly(
      chart_df,
      x = ~predicted_label,
      type = "histogram",
      marker = list(color = "#6366F1")
    ) %>%
      layout(
        title = "Prediction Direction Distribution",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        xaxis = list(title = "Predicted Direction"),
        yaxis = list(title = "Count", gridcolor = "#E2E8F0")
      )
  })
}

shinyApp(ui, server)
