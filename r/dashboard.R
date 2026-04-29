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
preds <- preds %>% mutate(
  Date = as.Date(Date),
  actual = as.integer(actual),
  pred_glm = as.integer(pred_glm),
  pred_rf = as.integer(pred_rf),
  pred_linreg = if ("pred_linreg" %in% colnames(.)) as.integer(pred_linreg) else 0L
)
featured <- featured %>% mutate(Date = as.Date(Date))
ticker_col <- if ("ticker" %in% colnames(featured)) "ticker" else NULL
ticker_choices <- if (!is.null(ticker_col)) sort(unique(as.character(featured[[ticker_col]]))) else character(0)
default_ticker <- if (length(ticker_choices) > 0) ticker_choices[[1]] else NULL

latest_close <- dplyr::last(featured$Close)
close_5d_ago <- dplyr::lag(featured$Close, 5) %>% dplyr::last()
recent_return <- ifelse(is.na(close_5d_ago) || close_5d_ago == 0, NA_real_, (latest_close / close_5d_ago) - 1)
best_row <- scores %>% arrange(desc(accuracy)) %>% slice(1)
last_data_date <- max(featured$Date, na.rm = TRUE)
default_forecast_date <- as.Date(last_data_date + 7)
model_choices <- c("Random Forest" = "pred_rf", "Logistic Regression" = "pred_glm")
if ("pred_linreg" %in% colnames(preds)) {
  model_choices <- c(model_choices, "Linear Regression" = "pred_linreg")
}

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
      selectInput("pred_model", "Prediction Model", choices = model_choices, selected = "pred_rf"),
      if (length(ticker_choices) > 0) {
        selectInput("ticker_filter", "Stock (Ticker)", choices = ticker_choices, selected = default_ticker)
      },
      sliderInput("lookback_days", "Price Chart Lookback (days)", min = 120, max = 2000, value = 600, step = 30),
      dateInput("forecast_date", "Future Date for Price Prediction", value = default_forecast_date, min = last_data_date),
      uiOutput("forecast_text"),
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
  business_days_between <- function(start_date, end_date) {
    if (is.na(start_date) || is.na(end_date) || end_date <= start_date) return(0L)
    all_days <- seq(as.Date(start_date + 1), as.Date(end_date), by = "day")
    if (length(all_days) == 0) return(0L)
    as.integer(sum(weekdays(all_days) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")))
  }

  featured_view <- reactive({
    if (!is.null(ticker_col) && !is.null(input$ticker_filter) && nzchar(input$ticker_filter)) {
      featured %>% filter(.data[[ticker_col]] == input$ticker_filter)
    } else {
      featured
    }
  })

  filtered_price <- reactive({
    p <- featured_view()
    cutoff <- max(p$Date, na.rm = TRUE) - input$lookback_days
    p %>% filter(Date >= cutoff)
  })

  output$pricePlot <- renderPlotly({
    p <- filtered_price()
    p_full <- featured_view() %>% arrange(Date)
    target_date <- as.Date(input$forecast_date)
    anchor_date <- max(p_full$Date, na.rm = TRUE)
    horizon_days <- business_days_between(anchor_date, target_date)

    recent_returns <- tail(p_full$return_1d[is.finite(p_full$return_1d)], 30)
    base_daily <- if (length(recent_returns) > 0) mean(recent_returns) else 0
    model_bias <- mean(tail(preds[[input$pred_model]], 120), na.rm = TRUE)
    adjusted_daily <- max(min(base_daily + ((model_bias - 0.5) * 0.0025), 0.08), -0.08)

    future_dates <- seq(anchor_date + 1, target_date, by = "day")
    future_dates <- future_dates[weekdays(future_dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
    forecast_path <- tibble(Date = as.Date(character()), Close = numeric())
    if (horizon_days > 0 && length(future_dates) > 0) {
      last_close_local <- dplyr::last(p_full$Close)
      mult <- (1 + adjusted_daily) ^ seq_along(future_dates)
      forecast_path <- tibble(Date = future_dates, Close = last_close_local * mult)
    }

    fig <- plot_ly(
      p,
      x = ~Date, y = ~Close,
      type = "scatter", mode = "lines",
      name = "Close",
      line = list(color = "#2563EB", width = 2.2)
    )
    if (nrow(forecast_path) > 0) {
      fig <- fig %>% add_trace(
        data = forecast_path,
        x = ~Date, y = ~Close,
        type = "scatter", mode = "lines",
        name = "Forecast",
        line = list(color = "#EF4444", width = 2, dash = "dash")
      )
    }
    fig %>% layout(
      title = if (!is.null(input$ticker_filter) && nzchar(input$ticker_filter)) {
        paste("Close Price Trend -", input$ticker_filter)
      } else {
        "Close Price Trend"
      },
      paper_bgcolor = "rgba(0,0,0,0)",
      plot_bgcolor = "rgba(0,0,0,0)",
      yaxis = list(title = "Price", gridcolor = "#E2E8F0"),
      xaxis = list(title = "Date", gridcolor = "#F1F5F9")
    )
  })

  output$forecast_text <- renderUI({
    p <- featured_view() %>% arrange(Date)
    if (nrow(p) == 0) {
      return(div(style = "margin-top:8px;color:#b91c1c;", "No data available for selected ticker."))
    }
    anchor_date <- max(p$Date, na.rm = TRUE)
    target_date <- as.Date(input$forecast_date)
    horizon_days <- business_days_between(anchor_date, target_date)

    if (is.na(target_date) || target_date <= anchor_date || horizon_days <= 0) {
      return(div(style = "margin-top:8px;color:#334155;", "Pick a date after the latest available date to generate a forecast."))
    }

    recent_returns <- tail(p$return_1d[is.finite(p$return_1d)], 30)
    base_daily <- if (length(recent_returns) > 0) mean(recent_returns) else 0
    model_bias <- mean(tail(preds[[input$pred_model]], 120), na.rm = TRUE)
    adjusted_daily <- max(min(base_daily + ((model_bias - 0.5) * 0.0025), 0.08), -0.08)

    last_close_local <- dplyr::last(p$Close)
    predicted_price <- last_close_local * ((1 + adjusted_daily) ^ horizon_days)
    projected_return <- ifelse(last_close_local == 0, 0, (predicted_price / last_close_local) - 1)

    ticker_label <- if (!is.null(input$ticker_filter) && nzchar(input$ticker_filter)) input$ticker_filter else "Selected stock"
    div(
      style = "margin-top:10px;padding:10px;background:#EEF2FF;border-radius:8px;color:#1E293B;",
      sprintf(
        "%s: %d business days ahead | Predicted price: $%.2f | Projected return: %s",
        ticker_label, horizon_days, predicted_price, scales::percent(projected_return, accuracy = 0.01)
      )
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
