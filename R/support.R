#' support functions for audrex
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile predict density
#' @importFrom utils head tail
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period is.Date as.duration
#' @importFrom modeest mlv1
#' @importFrom moments skewness kurtosis
#' @import purrr
#' @import imputeTS
#' @import abind
#' @import xgboost
#' @import ggplot2
#' @import tictoc
#' @import stringr

globalVariables(c("x_all", "y_all"))

reframe<-function(data, length)
{
  slice_list <- narray::split(data, along=2)
  reframed <- abind(map(slice_list, ~ t(apply(embed(.x, dimension=length), 1, rev))), along=3)
  return(reframed)
}

recursive_diff <- function(vector, deriv)
{
  head_value <- vector("numeric", deriv)
  tail_value <- vector("numeric", deriv)
  if(deriv==0){vector; head_value <- NULL; tail_value <- NULL}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector, head_value, tail_value)
  return(outcome)
}

invdiff <- function(vector, heads)
{
  if(is.null(heads)){return(vector)}
  if(!is.null(heads)){
    for(d in length(heads):1)
    {vector <- cumsum(c(heads[d], vector))}}
  return(vector)
}

eval_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  rmse <- sqrt(mean((actual - predicted)^2, na.rm = TRUE))
  mae <- mean(abs(actual - predicted), na.rm = TRUE)
  mdae <- median(abs(actual - predicted), na.rm = TRUE)
  mpe <- mean((actual - predicted)/actual, na.rm = TRUE)
  mape <- mean(abs(actual - predicted)/abs(actual), na.rm = TRUE)
  smape <- mean(abs(actual - predicted)/mean(c(abs(actual), abs(predicted)), na.rm = TRUE), na.rm = TRUE)

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mpe = mpe, mape = mape, smape = smape), 4)
  return(metrics)
}

minmax <- function(data, params = NULL, inverse = FALSE, by_col = FALSE)
{
  mm_norm <- function(x){(x - min(x, na.rm = TRUE))/(diff(range(x, na.rm = TRUE)))}
  inv_mm_norm <- function(x, range){x * diff(range) + min(range)}

  if(by_col == FALSE)
  {
    if(inverse == FALSE & is.null(params))
    {
      params <- range(data)
      transformed <- mm_norm(data)
      outcome <- list(params = params, transformed = transformed)
    }

    if(inverse == TRUE & !is.null(params))
    {
      inv_transformed <- inv_mm_norm(data, params)
      outcome <- list(inv_transformed = inv_transformed)
    }
  }

  if(by_col == TRUE)
  {

    if(inverse == FALSE & is.null(params))
    {
      if(is.matrix(data) || is.data.frame(data))
      {
        params <- apply(data, 2, range, na.rm = TRUE)
        transformed <- apply(data, 2, mm_norm)
      }

      if(is.vector(data))
      {
        params <- matrix(range(data, na.rm = TRUE), ncol = 1)
        transformed <- mm_norm(data)
      }

      outcome <- list(params = params, transformed = transformed)
    }

    if(inverse == TRUE & !is.null(params))
    {

      if(is.matrix(data) || is.data.frame(data))
      {
        inv_transformed <- mapply(function(x) inv_mm_norm(data[,x], params[,x]), x = 1:ncol(params))
      }

      if(is.vector(data))
      {
        inv_transformed <- inv_mm_norm(data, params)
      }

      outcome <- list(inv_transformed = inv_transformed)
    }
  }

  return(outcome)
}

sequential_kld <- function(matrix)
{
  n <- nrow(matrix)
  if(n == 1){return(c(NA, NA))}
  dens <- apply(matrix, 1, function(x) tryCatch(density(x[is.finite(x)], from = min(x, na.rm = TRUE), to = max(x, na.rm = TRUE)), error = function(e) NA))
  dens <- keep(dens, !is.na(dens))

  backward <- dens[-n]
  forward <- dens[-1]

  norm_backward <- map(backward, ~ .x$y/sum(.x$y))
  norm_forward <- map(forward, ~ .x$y/sum(.x$y))

  seq_kld <- map2_dbl(norm_forward, norm_backward, ~ sum(.x * log(.x/.y)))
  finite_index <- is.finite(seq_kld)
  if(all(finite_index == FALSE)){avg_seq_kld <- NA} else {avg_seq_kld <- round(mean(seq_kld[finite_index]), 3)}

  norm_end_dens <-  tryCatch(dens[[n]]$y/sum(dens[[n]]$y), error = function(e) NA)
  norm_init_dens <-  tryCatch(dens[[1]]$y/sum(dens[[1]]$y), error = function(e) NA)

  if(any(is.na(norm_end_dens) | is.na(norm_init_dens))){end_to_end_kld <- NA} else
  {
    end_to_end_kld <- norm_end_dens* log(norm_end_dens/norm_init_dens)
    finite_index <- is.finite(end_to_end_kld)
    if(all(finite_index == FALSE)){end_to_end_kld <- NA} else {end_to_end_kld <- round(sum(end_to_end_kld[finite_index]), 3)}
  }

  kld_stats <- c(avg_seq_kld, end_to_end_kld)

  return(kld_stats)
}

upside_probability <- function(matrix)
{
  n <- nrow(matrix)
  if(n == 1){return(c(NA, NA))}
  growths <- matrix[-1,]/matrix[-n,] - 1
  dens <- apply(growths, 1, function(x) tryCatch(density(x[is.finite(x)], from = min(x[is.finite(x)], na.rm = TRUE), to = max(x[is.finite(x)], na.rm = TRUE)), error = function(e) NA))
  dens <- keep(dens, !is.na(dens))
  avg_upp <- round(mean(map_dbl(dens, ~ sum(.x$y[.x$x>0])/sum(.x$y))), 3)
  end_growth <- matrix[n,]/matrix[1,] - 1
  end_to_end_dens <- tryCatch(density(end_growth[is.finite(end_growth)], from = min(end_growth[is.finite(end_growth)], na.rm = TRUE), to = max(end_growth[is.finite(end_growth)], na.rm = TRUE)), error = function(e) NA)
  if(!anyNA(end_to_end_dens)){last_to_first_upp <- round(sum(end_to_end_dens$y[end_to_end_dens$x>0])/sum(end_to_end_dens$y), 3)}
  if(anyNA(end_to_end_dens)){last_to_first_upp <- NA}
  upp_stats <- c(avg_upp, last_to_first_upp)
  return(upp_stats)
}

ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43",
                     label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{
  all_data <- data.frame("x_all" = c(x_hist, x_forcat), "y_all" = c(y_hist, y_forcat))
  forcat_data <- data.frame("x_forcat" = x_forcat, "y_forcat" = y_forcat)

  if(!is.null(lower) & !is.null(upper)){forcat_data$lower <- lower; forcat_data$upper <- upper}

  plot <- ggplot()+geom_line(data = all_data, aes(x = x_all, y = y_all), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes(x = x_forcat, ymin = lower, ymax = upper), alpha = 0.3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes(x = x_forcat, y = y_forcat), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = scales::number)
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}

xgb_ts <- function(data, target_label, regressors_labels, past, nrounds, patience, internal_holdout = 0.5, booster,
                   max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
                   verbose, reg, eval_metric)
{
  if(!is.data.frame(data) | !is.matrix(data)){data <- data.frame(data)}
  if(!all(c(target_label, regressors_labels) %in% colnames(data))){stop("missing target and/or regressors")}

  target_regressors_ts <- data[,c(target_label, regressors_labels),drop=FALSE]

  reg <- paste0("reg:", reg)

  if(booster=="gbtree")
  {
    param <- list(booster = booster, max_depth = max_depth, eta = eta,
                  gamma = gamma, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree = colsample_bytree,
                  objective = reg, eval_metric = eval_metric)
  }

  if(booster=="gblinear")
  {
    param <- list(booster = booster, eta = eta, lambda = lambda, alpha = alpha,
                  objective = reg, eval_metric = eval_metric)
  }

  target_regressors_reframed <- narray::split(reframe(target_regressors_ts, past + 1), along = 3)
  reframed <- Reduce(cbind, target_regressors_reframed)

  if(!is.null(regressors_labels))
  {
    discard_index <- seq(past + 1, ncol(reframed), past + 1)[-1]
    reframed <- reframed[, - discard_index, drop = FALSE]
  }

  train_index <- 1:round(nrow(reframed)*internal_holdout)
  valid_index <- setdiff(1:nrow(reframed), train_index)

  dtrain <- xgb.DMatrix(reframed[train_index,,drop=FALSE], label = reframed[train_index, past+1])
  dval <- xgb.DMatrix(reframed[valid_index,,drop=FALSE], label = reframed[valid_index, past+1])
  watchlist <- list(train = dtrain, eval = dval)

  final_model <- xgb.train(param, dtrain, watchlist = watchlist, nrounds = nrounds, early_stopping_rounds = patience, verbose = verbose)

  pred_fun <- function(new_data)
  {
    if(!is.data.frame(new_data) | !is.matrix(new_data)){new_data <- data.frame(new_data)}
    new_data <- tail(new_data, past)
    new_target_regressors_ts <- new_data[, c(target_label, regressors_labels), drop=FALSE]
    new_target_regressors_ts <- Reduce(c, narray::split(new_target_regressors_ts, along = 2))
    dim(new_target_regressors_ts) <- c(1, past * length(c(target_label, regressors_labels)))
    prediction <- predict(final_model, new_target_regressors_ts)
    return(prediction)
  }

  outcome <- list(pred_fun = pred_fun)
  return(outcome)
}


xgb_forecast <- function(data, targets, past, future, deriv, ci, n_windows, internal_holdout, nrounds, patience,
                         booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
                         verbose, reg, eval_metric, starting_date, time_unit, dbreak, minmax_model, orig, all_positive_check, all_negative_check)
{
  tic.clearlog()
  tic("time")

  data <- data[, targets, drop = FALSE]

  n_feat <- ncol(data)
  n_length <- nrow(data)

  window_index <- sort(c(rep(1, n_length%%n_windows), rep(1:n_windows, floor(n_length/n_windows))))

  ###TRAIN-TEST MODEL
  training_error_in_window <- vector("list", n_windows)
  testing_error_in_window <- vector("list", n_windows - 1)###THE LAST ONE TESTING IS THE PREDICTION ON NEW DATA
  raw_error_samples_in_window <- vector("list", n_windows - 1)


  for(w in 1:n_windows)
  {
    train_index <- c(1:n_length)[window_index <= w]
    train_length <- length(train_index)
    if(w != n_windows){test_index <- c(1:n_length)[window_index == w+1]; test_length <- length(test_index)}

    train_data <- data[train_index,,drop=FALSE]
    if(w != n_windows){test_data <- data[test_index,,drop=FALSE]}

    train_diff_models <- map2(train_data, deriv, ~ recursive_diff(.x, .y))
    train_data <- as.data.frame(lapply(map(train_diff_models, ~.x[[1]]), tail, n = train_length - max(deriv)))
    #train_diff_models <- purrr::map(train_data, ~ recursive_diff(.x, deriv))
    #train_data <- as.data.frame(purrr::map(train_diff_models, ~ .x[[1]]))

    if(w != n_windows)
    {
      test_diff_models <- map2(test_data, deriv, ~ recursive_diff(.x, .y))
      test_data <- as.data.frame(lapply(map(test_diff_models, ~.x[[1]]), tail, n = test_length - max(deriv)))
    }

    ts_models <- purrr::map(targets, ~ xgb_ts(train_data, target_label = .x, regressors_labels = if(n_feat == 1){NULL} else {setdiff(targets, .x)},
                                              past - max(deriv), nrounds, patience, internal_holdout, booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
                                              verbose, reg, eval_metric))

    ###PREDICTION SEEDS
    pred_funs <- purrr::map(ts_models, ~ .x$pred_fun)

    step_fun <- function(new_data, future)
    {
      for(n in 1:future){new_data <- rbind(new_data, purrr::map_dbl(pred_funs, ~ .x(new_data)))}
      return(as.data.frame(tail(new_data, future)))
    }

    if(w == n_windows)
    {
      new_data <- tail(train_data, past)###NEW DATA COMES FROM THE TAIL OF TRAIN SET IN THE NEW WINDOWED VALIDATION CYCLE
      predictions <- narray::split(step_fun(new_data, future = future), along = 2)

      tails <- purrr::map(train_diff_models, ~ .x[[3]])
      predictions <- purrr::map2(predictions, tails, ~ tail(invdiff(.x, .y), future))

      if(!is.null(minmax_model)){predictions <- map2(predictions, narray::split(minmax_model$params, along = 2), ~ unlist(minmax(.x, .y, inverse = TRUE, by_col = FALSE)))}
    }

    ###TRAIN ERROR ESTIMATION
    training_error <- NULL

    train_ground <- head(train_index, -(past + future - 1))
    ts_sequence_error <- vector("list", length=length(train_ground))
    one_step_error <- vector("list", length=length(train_ground))

    for(i in 1:length(train_ground))
    {
      sample_ground <- data[train_ground[i]:(train_ground[i] + past - 1),,drop=FALSE]
      sample_step <- data[(train_ground[i] + past):(train_ground[i] + past + future - 1),,drop=FALSE]

      sample_diff_models <- map2(sample_ground, deriv, ~ recursive_diff(.x, .y))
      sample_ground <- as.data.frame(lapply(map(sample_diff_models, ~.x[[1]]), tail, n = nrow(sample_ground) - max(deriv)))

      sample_pred <- step_fun(sample_ground, future)

      sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
      sample_pred <- as.data.frame(purrr::map2(sample_pred, sample_tails, ~ tail(invdiff(.x, .y), future)))

      sample_step <- narray::split(sample_step, along = 2)
      sample_pred <- narray::split(sample_pred, along = 2)

      ############################BACKTRANSF
      if(!is.null(minmax_model))
      {
        sample_step <- map2(sample_step, narray::split(minmax_model$params, along = 2), ~ unlist(minmax(.x, .y, inverse = TRUE, by_col = FALSE)))
        sample_pred <- map2(sample_pred, narray::split(minmax_model$params, along = 2), ~ unlist(minmax(.x, .y, inverse = TRUE, by_col = FALSE)))
      }

      one_step_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x[1], .y[1]))
      ts_sequence_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x, .y))
    }

    ts_sequence_error <- purrr::transpose(ts_sequence_error)
    ts_sequence_error <- purrr::map(ts_sequence_error, ~Reduce(rbind, .x))
    ts_sequence_error <- purrr::map(ts_sequence_error, ~ apply(.x, 2, mean))

    one_step_error <- purrr::transpose(one_step_error)
    one_step_error <- purrr::map(one_step_error, ~Reduce(rbind, .x))
    one_step_error <- purrr::map(one_step_error, ~ apply(.x, 2, mean))

    training_error <- purrr::transpose(list(one_step_error = one_step_error, ts_sequence_error = ts_sequence_error))
    training_error <- purrr::map(training_error, ~ Reduce(rbind, .x))
    training_error <- purrr::map(training_error, ~ {rownames(.x) <- c("one_step_error", "sequence_error"); return(.x)})
    training_error_in_window[[w]] <- training_error

    ###TEST ERROR ESTIMATION
    if(w != n_windows){
      test_ground <- head(test_index, -(past + future - 1))
      raw_error_samples <- vector("list", length=length(test_ground))
      ts_sequence_error <- vector("list", length=length(test_ground))
      one_step_error <- vector("list", length=length(test_ground))

      for(i in 1:length(test_ground))
      {
        sample_ground <- data[test_ground[i]:(test_ground[i] + past - 1),,drop=FALSE]
        sample_step <- data[(test_ground[i] + past):(test_ground[i] + past + future - 1),,drop=FALSE]

        sample_diff_models <- map2(sample_ground, deriv, ~ recursive_diff(.x, .y))
        sample_ground <- as.data.frame(lapply(map(sample_diff_models, ~.x[[1]]), tail, n = nrow(sample_ground) - max(deriv)))

        sample_pred <- step_fun(sample_ground, future)

        sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
        sample_pred <- as.data.frame(purrr::map2(sample_pred, sample_tails, ~ tail(invdiff(.x, .y), future)))

        sample_step <- narray::split(sample_step, along = 2)
        sample_pred <- narray::split(sample_pred, along = 2)

        ############################BACKTRANSF
        if(!is.null(minmax_model))
        {
          sample_step <- map2(sample_step, narray::split(minmax_model$params, along = 2), ~ unlist(minmax(.x, .y, inverse = TRUE, by_col = FALSE)))
          sample_pred <- map2(sample_pred, narray::split(minmax_model$params, along = 2), ~ unlist(minmax(.x, .y, inverse = TRUE, by_col = FALSE)))
        }

        raw_error_samples[[i]] <- map2(sample_step, sample_pred, ~ .x -.y)
        one_step_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x[1], .y[1]))
        ts_sequence_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x, .y))
      }

      raw_error_samples <- purrr::transpose(raw_error_samples)
      raw_error_samples <- purrr::map(raw_error_samples, ~Reduce(rbind, .x))
      raw_error_samples_in_window[[w]] <- raw_error_samples

      ts_sequence_error <- purrr::transpose(ts_sequence_error)
      ts_sequence_error <- purrr::map(ts_sequence_error, ~Reduce(rbind, .x))
      ts_sequence_error <- purrr::map(ts_sequence_error, ~ apply(.x, 2, mean))

      one_step_error <- purrr::transpose(one_step_error)
      one_step_error <- purrr::map(one_step_error, ~Reduce(rbind, .x))
      one_step_error <- purrr::map(one_step_error, ~ apply(.x, 2, mean))

      testing_error <- purrr::transpose(list(one_step_error = one_step_error, ts_sequence_error = ts_sequence_error))
      testing_error <- purrr::map(testing_error, ~ Reduce(rbind, .x))
      testing_error <- purrr::map(testing_error, ~ {rownames(.x) <- c("one_step_error", "sequence_error"); return(.x)})
      testing_error_in_window[[w]] <- testing_error
    }
  }

  ###ERROR INTEGRATION
  training_error <- map(transpose(training_error_in_window[-n_windows]), ~ Reduce('+', .x)/(n_windows - 1))
  testing_error <- map(transpose(testing_error_in_window), ~ Reduce('+', .x)/(n_windows - 1))
  raw_error_samples <- map(transpose(raw_error_samples_in_window), ~ Reduce(rbind, .x))

  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  q_names <- paste0("q", quants * 100)
  integrated_pred <- pmap(list(predictions, raw_error_samples), ~ t(mapply(function(t) ..1[t] + sample(..2[, t], size = 1000, replace = TRUE), t = 1:future)))

  if(any(all_positive_check)){integrated_pred <- map_if(integrated_pred, all_positive_check, ~ {.x[.x < 0] <- 0; return(.x)})}
  if(any(all_negative_check)){integrated_pred <- map_if(integrated_pred, all_negative_check, ~ {.x[.x > 0] <- 0; return(.x)})}

  quantile_predictor <- as_mapper(~t(apply(.x, 1, function(x){round(c(min(x, na.rm = TRUE), quantile(x, probs = quants, na.rm = TRUE), max(x, na.rm = TRUE), mean(x, na.rm = TRUE), sd(x, na.rm = TRUE), suppressWarnings(mlv1(x, method = "shorth", na.rm = TRUE)), skewness(x, na.rm=TRUE), kurtosis(x, na.rm=TRUE)), 3)})))
  predictions <- purrr::map(integrated_pred, quantile_predictor)
  predictions <- purrr::map(predictions, ~ {rownames(.x) <- NULL; colnames(.x) <- c("min", q_names, "max", "mean", "sd", "mode", "skewness", "kurtosis"); return(.x)})
  names(predictions) <- targets

  ###PREDICTION STATISTICS
  avg_iqr_to_range <- round(map_dbl(predictions, ~ mean((.x[,"q75"] - .x[,"q25"])/(.x[,"max"] - .x[,"min"]))), 3)
  last_to_first_iqr <- round(map_dbl(predictions, ~ (.x[future,"q75"] - .x[future,"q25"])/(.x[1,"q75"] - .x[1,"q25"])), 3)
  kld_stats <- as.data.frame(map(integrated_pred, ~ sequential_kld(.x)))
  upp_stats <- as.data.frame(map(integrated_pred, ~ upside_probability(.x)))

  pred_stats <- as.data.frame(rbind(avg_iqr_to_range, last_to_first_iqr, kld_stats, upp_stats))###WRAPPED AS DF TO AVOID TIBBLE WARNING IN FOLLOWING LINE
  rownames(pred_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio", "avg_kl_divergence", "terminal_kl_divergence", "avg_upside_prob", "terminal_upside_prob")

  ###SETTING DATES
  if(lubridate::is.Date(starting_date) && !is.null(time_unit))
  {
    time_unit <- as.duration(time_unit)
    ts_start_date <- starting_date
    ts_end_date <- as.Date(ts_start_date + time_unit * n_length)
    pred_start_date <- as.Date(ts_end_date + time_unit)
    pred_end_date <- as.Date(pred_start_date + time_unit * future)

    dates <- map(1:n_feat, ~ seq.Date(ts_start_date, ts_end_date, length.out = n_length))
    pred_dates <- map(1:n_feat, ~ seq.Date(pred_start_date, pred_end_date, length.out = future))

    predictions <- map2(predictions, pred_dates, ~ as.data.frame(cbind(dates=.y, .x)))
    predictions <- map(predictions, ~ {.x$dates <- as.Date(.x$dates, origin = "1970-01-01"); return(.x)})
  }

  if(!lubridate::is.Date(starting_date) || is.null(time_unit))
  {
    dates <- 1:n_length
    dates <- replicate(n_feat, dates, simplify = FALSE)
    pred_dates <- (n_length+1):(n_length+future)
    pred_dates <- replicate(n_feat, pred_dates, simplify = FALSE)
  }

  ###PREDICTION PLOT
  lower_name <- paste0("q", ((1-ci)/2) * 100)
  upper_name <- paste0("q", (ci+(1-ci)/2) * 100)

  plot <- pmap(list(orig, predictions, targets, dates, pred_dates), ~ ts_graph(x_hist = ..4, y_hist = ..1, x_forcat = ..5, y_forcat = ..2[, "q50"],
                                                                               lower = ..2[, lower_name], upper = ..2[, upper_name], label_x = paste0("Extreme Gradient Boosting Time Series Analysis (past = ", past ,", future = ", future,")"),
                                                                               label_y = paste0(str_to_title(..3), " Values"), dbreak = dbreak))

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(training_error = training_error, testing_error = testing_error, predictions = predictions, pred_stats = pred_stats, integrated_pred = integrated_pred, plot = plot)
  return(outcome)
}
