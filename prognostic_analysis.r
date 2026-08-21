library(survival)
library(survminer)
library(dplyr)
library(readr)
library(mRMRe)
library(timeROC)
library(ggplot2)


DATA_DIR <- "data"
RESULTS_DIR <- "results"

if (!dir.exists(RESULTS_DIR)) {
  dir.create(
    RESULTS_DIR,
    recursive = TRUE
  )
}


cohort_files <- list(
  Training = file.path(
    DATA_DIR,
    "training_data.csv"
  ),
  External_1 = file.path(
    DATA_DIR,
    "external_validation_1.csv"
  ),
  External_2 = file.path(
    DATA_DIR,
    "external_validation_2.csv"
  )
)


candidate_feature_file <- file.path(
  DATA_DIR,
  "candidate_features.csv"
)


cohort_endpoints <- list(
  Training = list(
    time = "dfs_day",
    event = "dfs_e"
  ),
  External_1 = list(
    time = "dfs_day",
    event = "dfs_e"
  ),
  External_2 = list(
    time = "PFI.time",
    event = "PFI"
  )
)


all_data <- lapply(
  names(cohort_files),
  function(cohort_name) {
    
    df <- read_csv(
      cohort_files[[cohort_name]],
      show_col_types = FALSE
    )
    
    if ("wsi_id" %in% names(df)) {
      df$wsi_id <- as.character(
        df$wsi_id
      )
    }
    
    return(df)
  }
)


names(all_data) <- names(
  cohort_files
)


train_hepaspot_model <- function(
    k_value = 20
) {
  
  train_df <- all_data[
    ["Training"]
  ]
  
  time_col <- cohort_endpoints[
    ["Training"]
  ]$time
  
  event_col <- cohort_endpoints[
    ["Training"]
  ]$event
  
  
  candidate_features <- read_csv(
    candidate_feature_file,
    show_col_types = FALSE
  ) %>%
    pull(feature)
  
  
  candidate_features <- intersect(
    candidate_features,
    names(train_df)
  )
  
  
  mrmr_input <- train_df[
    ,
    c(
      time_col,
      candidate_features
    )
  ]
  
  
  mrmr_input <- as.data.frame(
    mrmr_input
  )
  
  
  mrmr_input <- na.omit(
    mrmr_input
  )
  
  
  mrmr_result <- mRMR.ensemble(
    data = mRMR.data(
      data = mrmr_input
    ),
    target_indices = 1,
    solution_count = 1,
    feature_count = k_value
  )
  
  
  selected_features <- names(
    mrmr_input
  )[
    as.vector(
      solutions(
        mrmr_result
      )[[1]]
    )
  ]
  
  
  message(
    "Selected features: ",
    paste(
      selected_features,
      collapse = ", "
    )
  )
  
  
  formula_string <- paste0(
    "Surv(",
    time_col,
    ", ",
    event_col,
    ") ~ ",
    paste(
      paste0(
        "`",
        selected_features,
        "`"
      ),
      collapse = " + "
    )
  )
  
  
  model <- coxph(
    as.formula(
      formula_string
    ),
    data = train_df,
    x = TRUE
  )
  
  
  valid_train_df <- train_df %>%
    filter(
      !is.na(
        .data[[time_col]]
      ),
      !is.na(
        .data[[event_col]]
      )
    )
  
  
  valid_train_df <- valid_train_df[
    complete.cases(
      valid_train_df[
        ,
        selected_features
      ]
    ),
  ]
  
  
  training_risk_score <- predict(
    model,
    newdata = valid_train_df,
    type = "lp"
  )
  
  
  cutoff <- median(
    training_risk_score,
    na.rm = TRUE
  )
  
  
  return(
    list(
      model = model,
      features = selected_features,
      cutoff = cutoff
    )
  )
}


hepaspot_model <- train_hepaspot_model(
  k_value = 20
)


get_evaluation_data <- function(
    cohort_name,
    model_object
) {
  
  df <- all_data[
    [cohort_name]
  ]
  
  
  time_col <- cohort_endpoints[
    [cohort_name]
  ]$time
  
  event_col <- cohort_endpoints[
    [cohort_name]
  ]$event
  
  
  df <- df %>%
    filter(
      !is.na(
        .data[[time_col]]
      ),
      !is.na(
        .data[[event_col]]
      )
    )
  
  
  df <- df[
    complete.cases(
      df[
        ,
        model_object$features
      ]
    ),
  ]
  
  
  df$time_col <- as.numeric(
    df[[time_col]]
  )
  
  df$event_col <- as.numeric(
    df[[event_col]]
  )
  
  
  df$risk_score <- predict(
    model_object$model,
    newdata = df,
    type = "lp"
  )
  
  
  return(df)
}


calculate_cindex <- function(
    model_object,
    cohort_name
) {
  
  df <- get_evaluation_data(
    cohort_name,
    model_object
  )
  
  
  cindex_result <- concordance(
    Surv(
      df$time_col,
      df$event_col
    ) ~ df$risk_score,
    reverse = TRUE
  )
  
  
  c_index <- cindex_result$concordance
  
  se <- sqrt(
    cindex_result$var
  )
  
  
  lower_ci <- c_index - 1.96 * se
  upper_ci <- c_index + 1.96 * se
  
  
  return(
    data.frame(
      Cohort = cohort_name,
      C_index = c_index,
      C_index_lower = lower_ci,
      C_index_upper = upper_ci
    )
  )
}


calculate_time_auc <- function(
    model_object,
    cohort_name,
    time_points = c(
      365,
      730
    )
) {
  
  df <- get_evaluation_data(
    cohort_name,
    model_object
  )
  
  
  roc_result <- timeROC(
    T = df$time_col,
    delta = df$event_col,
    marker = df$risk_score,
    cause = 1,
    weighting = "marginal",
    times = time_points,
    iid = TRUE
  )
  
  
  get_standard_error <- function(
    result,
    index
  ) {
    
    if (
      !is.null(
        result$inference$std_error
      )
    ) {
      return(
        result$inference$std_error[
          index
        ]
      )
    }
    
    
    if (
      !is.null(
        result$inference$vect_sd_1
      )
    ) {
      return(
        result$inference$vect_sd_1[
          index
        ]
      )
    }
    
    
    return(NA)
  }
  
  
  results <- lapply(
    seq_along(
      time_points
    ),
    function(index) {
      
      auc <- roc_result$AUC[
        index
      ]
      
      se <- get_standard_error(
        roc_result,
        index
      )
      
      
      if (
        is.na(se)
      ) {
        
        lower_ci <- NA
        upper_ci <- NA
        
      } else {
        
        lower_ci <- max(
          0,
          auc - 1.96 * se
        )
        
        upper_ci <- min(
          1,
          auc + 1.96 * se
        )
      }
      
      
      data.frame(
        Cohort = cohort_name,
        Time_days = time_points[
          index
        ],
        AUC = auc,
        AUC_lower = lower_ci,
        AUC_upper = upper_ci
      )
    }
  )
  
  
  return(
    bind_rows(
      results
    )
  )
}


plot_km <- function(
    model_object,
    cohort_name
) {
  
  df <- get_evaluation_data(
    cohort_name,
    model_object
  )
  
  
  df$event_5y <- ifelse(
    df$time_col > 1825,
    0,
    df$event_col
  )
  
  
  df$time_5y <- ifelse(
    df$time_col > 1825,
    1825,
    df$time_col
  )
  
  
  df$risk_group <- factor(
    ifelse(
      df$risk_score >
        model_object$cutoff,
      "High risk",
      "Low risk"
    ),
    levels = c(
      "High risk",
      "Low risk"
    )
  )
  
  
  km_fit <- survfit(
    Surv(
      time_5y,
      event_5y
    ) ~ risk_group,
    data = df
  )
  
  
  km_plot <- ggsurvplot(
    km_fit,
    data = df,
    pval = TRUE,
    conf.int = FALSE,
    risk.table = TRUE,
    risk.table.title = "Number at risk",
    risk.table.y.text = TRUE,
    xlab = "Time (days)",
    ylab = "Survival probability",
    xlim = c(
      0,
      1825
    ),
    break.time.by = 365,
    censor.size = 2,
    title = cohort_name,
    legend.title = "",
    legend.labs = c(
      "High risk",
      "Low risk"
    )
  )
  
  
  return(
    km_plot
  )
}


cohort_names <- c(
  "Training",
  "External_1",
  "External_2"
)


cindex_results <- bind_rows(
  lapply(
    cohort_names,
    function(cohort_name) {
      
      calculate_cindex(
        hepaspot_model,
        cohort_name
      )
    }
  )
)


auc_results <- bind_rows(
  lapply(
    cohort_names,
    function(cohort_name) {
      
      calculate_time_auc(
        hepaspot_model,
        cohort_name,
        time_points = c(
          365,
          730
        )
      )
    }
  )
)


write_csv(
  cindex_results,
  file.path(
    RESULTS_DIR,
    "hepaspot_cindex.csv"
  )
)


write_csv(
  auc_results,
  file.path(
    RESULTS_DIR,
    "hepaspot_time_auc.csv"
  )
)


selected_feature_table <- data.frame(
  Feature = hepaspot_model$features
)


write_csv(
  selected_feature_table,
  file.path(
    RESULTS_DIR,
    "hepaspot_selected_features.csv"
  )
)


writeLines(
  as.character(
    hepaspot_model$cutoff
  ),
  file.path(
    RESULTS_DIR,
    "hepaspot_training_cutoff.txt"
  )
)


print(
  cindex_results
)

print(
  auc_results
)


km_training <- plot_km(
  hepaspot_model,
  "Training"
)

km_external_1 <- plot_km(
  hepaspot_model,
  "External_1"
)

km_external_2 <- plot_km(
  hepaspot_model,
  "External_2"
)


print(
  km_training
)

print(
  km_external_1
)

print(
  km_external_2
)