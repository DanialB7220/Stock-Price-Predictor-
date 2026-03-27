suppressPackageStartupMessages({
  library(shiny)
  library(plotly)
  library(readr)
  library(dplyr)
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

ui <- fluidPage(
  titlePanel("Stock Trend Prediction Dashboard (R Shiny)"),
  sidebarLayout(
    sidebarPanel(
      selectInput("pred_model", "Prediction Column", choices = c("pred_glm", "pred_rf"), selected = "pred_rf")
    ),
    mainPanel(
      plotlyOutput("pricePlot"),
      plotlyOutput("scorePlot"),
      plotlyOutput("predPlot")
    )
  )
)

server <- function(input, output, session) {
  output$pricePlot <- renderPlotly({
    plot_ly(featured, x = ~Date, y = ~Close, type = "scatter", mode = "lines") %>%
      layout(title = "Close Price Over Time")
  })

  output$scorePlot <- renderPlotly({
    plot_ly(scores, x = ~model, y = ~accuracy, type = "bar") %>%
      layout(title = "R Model Accuracy Comparison")
  })

  output$predPlot <- renderPlotly({
    plot_ly(preds, x = as.formula(paste0("~", input$pred_model)), type = "histogram") %>%
      layout(title = paste("Prediction Distribution:", input$pred_model))
  })
}

shinyApp(ui, server)
