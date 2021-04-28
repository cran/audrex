#' support functions for audrex
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile predict
#' @importFrom utils head tail
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period
#' @importFrom bizdays create.calendar add.bizdays bizseq
#' @import purrr
#' @import imputeTS
#' @import abind
#' @import xgboost
#' @import ggplot2
#' @import tictoc
#' @import stringr

globalVariables(c("x_all", "y_all", "shift"))

make_alist <- function(args) {
  res <- replicate(length(args), substitute())
  names(res) <- args
  res}

make_function <- function(args, body, env = parent.frame()) {
  args <- as.pairlist(args)
  eval(call("function", args, body), env)}


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
  if(deriv==0){vector}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector, head_value, tail_value)
  return(outcome)
}

invdiff <- function(vector, heads)
{
  for(d in length(heads):1)
  {vector <- cumsum(c(heads[d], vector))}
  return(vector)
}

eval_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mdae <- median(abs(actual - predicted))
  mpe <- mean((actual - predicted)/actual)
  mape <- mean(abs(actual - predicted)/abs(actual))
  smape <- mean(abs(actual - predicted)/mean(c(abs(actual), abs(predicted))))

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mpe = mpe, mape = mape, smape = smape), 4)
  return(metrics)
}

pred_statistics <- function(list_of_predictions, reference_points, ecdf_by_time, future, targets)
{
  iqr_to_range <- round(map_dbl(list_of_predictions, ~ mean((.x[,"q75"] - .x[,"q25"])/(.x[,"max"] - .x[,"min"]))), 3)
  iqr_ratio <- round(map_dbl(list_of_predictions, ~ (.x[future,"q75"] - .x[future,"q25"])/(.x[1,"q75"] - .x[1,"q25"])), 3)
  upside_prob <- unlist(map2(reference_points, ecdf_by_time, ~ mean(mapply(function(f) (1 - ..2[[f]](..1)), f = 1:future))))
  pred_stats <- round(rbind(iqr_to_range, iqr_ratio, upside_prob), 4)
  rownames(pred_stats) <- c("iqr_to_range", "iqr_ratio", "upside_prob")
  colnames(pred_stats) <- targets

  return(pred_stats)
}

ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43",
                     label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{
  all_data <- data.frame(x_all = c(x_hist, x_forcat), y_all = c(y_hist, y_forcat))
  forcat_data <- data.frame(x_forcat = x_forcat, y_forcat = y_forcat)

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

xgb_ts <- function(data, target_label, regressors_labels, past, nrounds, patience, holdout, booster,
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

  train_index <- 1:round(nrow(reframed)*(1 - holdout))
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


xgb_forecast <- function(data, targets, past, deriv = 0, future, ci = 0.8, holdout = 0.3, nrounds = 100, patience = 10,
                         booster = "gbtree", max_depth = 4, eta = 0.3, gamma = 1, min_child_weight = 1, subsample = 1, colsample_bytree = 1, lambda = 0, alpha = 0,
                         verbose = FALSE, reg = "squarederror", eval_metric = "rmse", starting_date = NULL, dbreak = NULL, days_off = NULL, include_training_error = FALSE, min_set = 30)
{
  tic.clearlog()
  tic("time")

  data <- data[, targets, drop = FALSE]
  n_feat <- ncol(data)
  n_length <- nrow(data)

  ###TRAIN-TEST MODEL
  train_index <- 1:round(n_length * (1 - holdout))
  test_index <- setdiff(1:n_length, train_index)
  train_data <- data[train_index,,drop=FALSE]
  test_data <- data[test_index,,drop=FALSE]

  if((length(train_index) - min_set + 1) < (past + future) | (length(test_index) - min_set + 1) < (past + future))
  {
    message("\ninsufficient data points for the forecasting horizon\n")
    past <- min(c(length(train_index), length(test_index))) - future - (min_set - 1)
    if(past >= 1){message("setting past to max possible value,", past,"points, for a minimal", min_set ,"validation sequences\n")}
    if(past < 1){stop("need to reset both past and future parameters or adjust holdout\n")}
  }

  if(deriv > 0)
  {
    train_diff_models <- purrr::map(train_data, ~ recursive_diff(.x, deriv))
    train_data <- as.data.frame(purrr::map(train_diff_models, ~ .x[[1]]))

    test_diff_models <- purrr::map(test_data, ~ recursive_diff(.x, deriv))
    test_data <- as.data.frame(purrr::map(test_diff_models, ~ .x[[1]]))
  }

  ts_models <- purrr::map(targets, ~ xgb_ts(train_data, target_label = .x, regressors_labels = if(n_feat == 1){NULL} else {setdiff(targets, .x)},
                                            past - deriv, nrounds, patience, holdout, booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
                                            verbose, reg, eval_metric))

  ###PREDICTION SEEDS
  pred_funs <- purrr::map(ts_models, ~ .x$pred_fun)

  step_fun <- function(new_data, future)
  {
    for(n in 1:future){new_data <- rbind(new_data, purrr::map_dbl(pred_funs, ~ .x(new_data)))}
    return(as.data.frame(tail(new_data, future)))
  }

  new_data <- tail(test_data, past)###NEW DATA COMES FROM THE TAIL OF TEST SET
  predictions <- narray::split(step_fun(new_data, future = future), along = 2)

  if(deriv > 0)
  {
    tails <- purrr::map(test_diff_models, ~ .x[[3]])
    predictions <- purrr::map2(predictions, tails, ~ tail(invdiff(.x, .y), future))
  }

  ###TRAIN ERROR ESTIMATION
  training_error <- NULL
  if(include_training_error == TRUE)
  {
    train_ground <- head(train_index, -(past + future - 1))
    ts_sequence_error <- vector("list", length=length(train_ground))
    one_step_error <- vector("list", length=length(train_ground))

    for(i in 1:length(train_ground))
    {
      sample_ground <- data[train_ground[i]:(train_ground[i] + past - 1),,drop=FALSE]
      sample_step <- data[(train_ground[i] + past):(train_ground[i] + past + future - 1),,drop=FALSE]

      if(deriv > 0)
      {
        sample_diff_models <- purrr::map(sample_ground, ~ recursive_diff(.x, deriv))
        sample_ground <- as.data.frame(purrr::map(sample_diff_models, ~ .x[[1]]))
      }

      sample_pred <- step_fun(sample_ground, future)

      if(deriv > 0)
      {
        sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
        sample_pred <- tail(as.data.frame(purrr::map2(sample_pred, sample_tails, ~ invdiff(.x, .y))), future)
      }

      sample_step <- narray::split(sample_step, along = 2)
      sample_pred <- narray::split(sample_pred, along = 2)

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
  }

  ###TEST ERROR ESTIMATION
  test_ground <- head(test_index, -(past + future - 1))
  raw_error_samples <- vector("list", length=length(test_ground))
  ts_sequence_error <- vector("list", length=length(test_ground))
  one_step_error <- vector("list", length=length(test_ground))

  for(i in 1:length(test_ground))
  {
    sample_ground <- data[test_ground[i]:(test_ground[i] + past - 1),,drop=FALSE]
    sample_step <- data[(test_ground[i] + past):(test_ground[i] + past + future - 1),,drop=FALSE]

    if(deriv > 0)
    {
      sample_diff_models <- purrr::map(sample_ground, ~ recursive_diff(.x, deriv))
      sample_ground <- as.data.frame(purrr::map(sample_diff_models, ~ .x[[1]]))
    }

    sample_pred <- step_fun(sample_ground, future)

    if(deriv > 0)
    {
      sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
      sample_pred <- tail(as.data.frame(purrr::map2(sample_pred, sample_tails, ~ invdiff(.x, .y))), future)
    }

    sample_step <- narray::split(sample_step, along = 2)
    sample_pred <- narray::split(sample_pred, along = 2)

    raw_error_samples[[i]] <- map2(sample_step, sample_pred, ~ .x -.y)
    one_step_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x[1], .y[1]))
    ts_sequence_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x, .y))
  }

  raw_error_samples <- purrr::transpose(raw_error_samples)
  raw_error_samples <- purrr::map(raw_error_samples, ~Reduce(rbind, .x))
  error_means <- purrr::map(raw_error_samples, ~ apply(.x, 2, mean))
  error_sds <- purrr::map(raw_error_samples, ~ apply(.x, 2, sd))

  ts_sequence_error <- purrr::transpose(ts_sequence_error)
  ts_sequence_error <- purrr::map(ts_sequence_error, ~Reduce(rbind, .x))
  ts_sequence_error <- purrr::map(ts_sequence_error, ~ apply(.x, 2, mean))

  one_step_error <- purrr::transpose(one_step_error)
  one_step_error <- purrr::map(one_step_error, ~Reduce(rbind, .x))
  one_step_error <- purrr::map(one_step_error, ~ apply(.x, 2, mean))

  testing_error <- purrr::transpose(list(one_step_error = one_step_error, ts_sequence_error = ts_sequence_error))
  testing_error <- purrr::map(testing_error, ~ Reduce(rbind, .x))
  testing_error <- purrr::map(testing_error, ~ {rownames(.x) <- c("one_step_error", "sequence_error"); return(.x)})

  ###ERROR INTEGRATION
  error_means <- purrr::map(error_means, ~  t(as.data.frame(.x)))
  error_sds <- purrr::map(error_sds, ~  t(as.data.frame(.x)))

  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  q_names <- paste0("q", quants * 100)
  integrated_pred <- pmap(list(predictions, error_means, error_sds), ~ t(mapply(function(t) ..1[t] + rnorm(1000, mean = ..2[t], sd = ..3[t]), t = 1:future)))

  quantile_predictor <- as_mapper(~t(apply(.x, 1, function(x){c(min(x, na.rm = TRUE), quantile(x, probs = quants, na.rm = TRUE), max(x, na.rm = TRUE), mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))})))
  predictions <- purrr::map(integrated_pred, quantile_predictor)
  predictions <- purrr::map(predictions, ~ {rownames(.x) <- NULL; colnames(.x) <- c("min", q_names, "max", "mean", "sd"); return(.x)})
  names(predictions) <- targets

  ecdf_by_time <-  map(integrated_pred, ~ apply(.x, 1, ecdf))
  pred_stats <- pred_statistics(predictions, tail(data, 1), ecdf_by_time, future, targets)

  ###SETTING DATES
  if(!is.null(starting_date))
  {
    date_list <- map(shift, ~ seq.Date(as.Date(starting_date) + .x, as.Date(starting_date) + .x + n_length, length.out = n_length))
    dates <- date_list
    start_date <- map(date_list, ~ tail(.x, 1))
    mycal <- create.calendar(name="mycal", weekdays=days_off)
    end_day <- map(start_date, ~ add.bizdays(.x, future, cal=mycal))
    pred_dates <- map2(start_date, end_day, ~ tail(bizseq(.x, .y, cal=mycal), future))
    predictions <- map2(predictions, pred_dates, ~ as.data.frame(cbind(dates=.y, .x)))
    predictions <- map(predictions, ~ {.x$dates <- as.Date(.x$dates, origin = "1970-01-01"); return(.x)})
  }

  if(is.null(starting_date))
  {
    dates <- 1:n_length
    dates <- replicate(n_feat, dates, simplify = FALSE)
    pred_dates <- (n_length+1):(n_length+future)
    pred_dates <- replicate(n_feat, pred_dates, simplify = FALSE)
  }

  ###PREDICTION PLOT
  lower_name <- paste0("q", ((1-ci)/2) * 100)
  upper_name <- paste0("q", (ci+(1-ci)/2) * 100)

  plot <- pmap(list(data, predictions, targets, dates, pred_dates), ~ ts_graph(x_hist = ..4, y_hist = ..1, x_forcat = ..5, y_forcat = ..2[, "q50"],
                                                                               lower = ..2[, lower_name], upper = ..2[, upper_name], label_x = paste0("Extreme Gradient Boosting Machine Time Series Analysis (past = ", past ,", future = ", future,")"),
                                                                               label_y = paste0(str_to_title(..3), " Values"), dbreak = dbreak))

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(training_error = training_error, testing_error = testing_error, predictions = predictions, pred_stats = pred_stats, plot = plot)
  return(outcome)
}



#' support functions for audrex
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats lm median na.omit quantile predict
#' @importFrom utils head tail
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period
#' @importFrom bizdays create.calendar add.bizdays bizseq
#' @import purrr
#' @import imputeTS
#' @import abind
#' @import xgboost
#' @import ggplot2
#' @import tictoc
#' @import stringr

globalVariables(c("x_all", "y_all", "shift"))

make_alist <- function(args) {
  res <- replicate(length(args), substitute())
  names(res) <- args
  res}

make_function <- function(args, body, env = parent.frame()) {
  args <- as.pairlist(args)
  eval(call("function", args, body), env)}


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
  if(deriv==0){vector}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector, head_value, tail_value)
  return(outcome)
}

invdiff <- function(vector, heads)
{
  for(d in length(heads):1)
  {vector <- cumsum(c(heads[d], vector))}
  return(vector)
}

eval_metrics <- function(actual, predicted)
{
  actual <- unlist(actual)
  predicted <- unlist(predicted)
  if(length(actual) != length(predicted)){stop("different lengths")}

  rmse <- sqrt(mean((actual - predicted)^2))
  mae <- mean(abs(actual - predicted))
  mdae <- median(abs(actual - predicted))
  mpe <- mean((actual - predicted)/actual)
  mape <- mean(abs(actual - predicted)/abs(actual))
  smape <- mean(abs(actual - predicted)/mean(c(abs(actual), abs(predicted))))

  metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mpe = mpe, mape = mape, smape = smape), 4)
  return(metrics)
}

pred_statistics <- function(list_of_predictions, reference_points, ecdf_by_time, future, targets)
{
  iqr_to_range <- round(map_dbl(list_of_predictions, ~ mean((.x[,"q75"] - .x[,"q25"])/(.x[,"max"] - .x[,"min"]))), 3)
  iqr_ratio <- round(map_dbl(list_of_predictions, ~ (.x[future,"q75"] - .x[future,"q25"])/(.x[1,"q75"] - .x[1,"q25"])), 3)
  upside_prob <- unlist(map2(reference_points, ecdf_by_time, ~ mean(mapply(function(f) (1 - ..2[[f]](..1)), f = 1:future))))
  pred_stats <- round(rbind(iqr_to_range, iqr_ratio, upside_prob), 4)
  rownames(pred_stats) <- c("iqr_to_range", "iqr_ratio", "upside_prob")
  colnames(pred_stats) <- targets

  return(pred_stats)
}

ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43",
                     label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{
  all_data <- data.frame(x_all = c(x_hist, x_forcat), y_all = c(y_hist, y_forcat))
  forcat_data <- data.frame(x_forcat = x_forcat, y_forcat = y_forcat)

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

xgb_ts <- function(data, target_label, regressors_labels, past, nrounds, patience, holdout, booster,
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

  train_index <- 1:round(nrow(reframed)*(1 - holdout))
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


xgb_forecast <- function(data, targets, past, deriv = 0, future, ci = 0.8, holdout = 0.3, nrounds = 100, patience = 10,
                         booster = "gbtree", max_depth = 4, eta = 0.3, gamma = 1, min_child_weight = 1, subsample = 1, colsample_bytree = 1, lambda = 0, alpha = 0,
                         verbose = FALSE, reg = "squarederror", eval_metric = "rmse", starting_date = NULL, dbreak = NULL, days_off = NULL, include_training_error = FALSE, min_set = 30)
{
  tic.clearlog()
  tic("time")

  data <- data[, targets, drop = FALSE]
  n_feat <- ncol(data)
  n_length <- nrow(data)

  ###TRAIN-TEST MODEL
  train_index <- 1:round(n_length * (1 - holdout))
  test_index <- setdiff(1:n_length, train_index)
  train_data <- data[train_index,,drop=FALSE]
  test_data <- data[test_index,,drop=FALSE]

  if((length(train_index) - min_set + 1) < (past + future) | (length(test_index) - min_set + 1) < (past + future))
  {
    message("\ninsufficient data points for the forecasting horizon\n")
    past <- min(c(length(train_index), length(test_index))) - future - (min_set - 1)
    if(past >= 1){message("setting past to max possible value,", past,"points, for a minimal", min_set ,"validation sequences\n")}
    if(past < 1){stop("need to reset both past and future parameters or adjust holdout\n")}
  }

  if(deriv > 0)
  {
    train_diff_models <- purrr::map(train_data, ~ recursive_diff(.x, deriv))
    train_data <- as.data.frame(purrr::map(train_diff_models, ~ .x[[1]]))

    test_diff_models <- purrr::map(test_data, ~ recursive_diff(.x, deriv))
    test_data <- as.data.frame(purrr::map(test_diff_models, ~ .x[[1]]))
  }

  ts_models <- purrr::map(targets, ~ xgb_ts(train_data, target_label = .x, regressors_labels = if(n_feat == 1){NULL} else {setdiff(targets, .x)},
                                            past - deriv, nrounds, patience, holdout, booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
                                            verbose, reg, eval_metric))

  ###PREDICTION SEEDS
  pred_funs <- purrr::map(ts_models, ~ .x$pred_fun)

  step_fun <- function(new_data, future)
  {
    for(n in 1:future){new_data <- rbind(new_data, purrr::map_dbl(pred_funs, ~ .x(new_data)))}
    return(as.data.frame(tail(new_data, future)))
  }

  new_data <- tail(test_data, past)###NEW DATA COMES FROM THE TAIL OF TEST SET
  predictions <- narray::split(step_fun(new_data, future = future), along = 2)

  if(deriv > 0)
  {
    tails <- purrr::map(test_diff_models, ~ .x[[3]])
    predictions <- purrr::map2(predictions, tails, ~ tail(invdiff(.x, .y), future))
  }

  ###TRAIN ERROR ESTIMATION
  training_error <- NULL
  if(include_training_error == TRUE)
  {
    train_ground <- head(train_index, -(past + future - 1))
    ts_sequence_error <- vector("list", length=length(train_ground))
    one_step_error <- vector("list", length=length(train_ground))

    for(i in 1:length(train_ground))
    {
      sample_ground <- data[train_ground[i]:(train_ground[i] + past - 1),,drop=FALSE]
      sample_step <- data[(train_ground[i] + past):(train_ground[i] + past + future - 1),,drop=FALSE]

      if(deriv > 0)
      {
        sample_diff_models <- purrr::map(sample_ground, ~ recursive_diff(.x, deriv))
        sample_ground <- as.data.frame(purrr::map(sample_diff_models, ~ .x[[1]]))
      }

      sample_pred <- step_fun(sample_ground, future)

      if(deriv > 0)
      {
        sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
        sample_pred <- tail(as.data.frame(purrr::map2(sample_pred, sample_tails, ~ invdiff(.x, .y))), future)
      }

      sample_step <- narray::split(sample_step, along = 2)
      sample_pred <- narray::split(sample_pred, along = 2)

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
  }

  ###TEST ERROR ESTIMATION
  test_ground <- head(test_index, -(past + future - 1))
  raw_error_samples <- vector("list", length=length(test_ground))
  ts_sequence_error <- vector("list", length=length(test_ground))
  one_step_error <- vector("list", length=length(test_ground))

  for(i in 1:length(test_ground))
  {
    sample_ground <- data[test_ground[i]:(test_ground[i] + past - 1),,drop=FALSE]
    sample_step <- data[(test_ground[i] + past):(test_ground[i] + past + future - 1),,drop=FALSE]

    if(deriv > 0)
    {
      sample_diff_models <- purrr::map(sample_ground, ~ recursive_diff(.x, deriv))
      sample_ground <- as.data.frame(purrr::map(sample_diff_models, ~ .x[[1]]))
    }

    sample_pred <- step_fun(sample_ground, future)

    if(deriv > 0)
    {
      sample_tails <- purrr::map(sample_diff_models, ~ .x[[3]])
      sample_pred <- tail(as.data.frame(purrr::map2(sample_pred, sample_tails, ~ invdiff(.x, .y))), future)
    }

    sample_step <- narray::split(sample_step, along = 2)
    sample_pred <- narray::split(sample_pred, along = 2)

    raw_error_samples[[i]] <- map2(sample_step, sample_pred, ~ .x -.y)
    one_step_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x[1], .y[1]))
    ts_sequence_error[[i]] <- map2(sample_step, sample_pred, ~ eval_metrics(.x, .y))
  }

  raw_error_samples <- purrr::transpose(raw_error_samples)
  raw_error_samples <- purrr::map(raw_error_samples, ~Reduce(rbind, .x))
  error_means <- purrr::map(raw_error_samples, ~ apply(.x, 2, mean))
  error_sds <- purrr::map(raw_error_samples, ~ apply(.x, 2, sd))

  ts_sequence_error <- purrr::transpose(ts_sequence_error)
  ts_sequence_error <- purrr::map(ts_sequence_error, ~Reduce(rbind, .x))
  ts_sequence_error <- purrr::map(ts_sequence_error, ~ apply(.x, 2, mean))

  one_step_error <- purrr::transpose(one_step_error)
  one_step_error <- purrr::map(one_step_error, ~Reduce(rbind, .x))
  one_step_error <- purrr::map(one_step_error, ~ apply(.x, 2, mean))

  testing_error <- purrr::transpose(list(one_step_error = one_step_error, ts_sequence_error = ts_sequence_error))
  testing_error <- purrr::map(testing_error, ~ Reduce(rbind, .x))
  testing_error <- purrr::map(testing_error, ~ {rownames(.x) <- c("one_step_error", "sequence_error"); return(.x)})

  ###ERROR INTEGRATION
  error_means <- purrr::map(error_means, ~  t(as.data.frame(.x)))
  error_sds <- purrr::map(error_sds, ~  t(as.data.frame(.x)))

  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  q_names <- paste0("q", quants * 100)
  integrated_pred <- pmap(list(predictions, error_means, error_sds), ~ t(mapply(function(t) ..1[t] + rnorm(1000, mean = ..2[t], sd = ..3[t]), t = 1:future)))

  quantile_predictor <- as_mapper(~t(apply(.x, 1, function(x){c(min(x, na.rm = TRUE), quantile(x, probs = quants, na.rm = TRUE), max(x, na.rm = TRUE), mean(x, na.rm = TRUE), sd(x, na.rm = TRUE))})))
  predictions <- purrr::map(integrated_pred, quantile_predictor)
  predictions <- purrr::map(predictions, ~ {rownames(.x) <- NULL; colnames(.x) <- c("min", q_names, "max", "mean", "sd"); return(.x)})
  names(predictions) <- targets

  ecdf_by_time <-  map(integrated_pred, ~ apply(.x, 1, ecdf))
  pred_stats <- pred_statistics(predictions, tail(data, 1), ecdf_by_time, future, targets)

  ###SETTING DATES
  if(!is.null(starting_date))
  {
    date_list <- map(shift, ~ seq.Date(as.Date(starting_date) + .x, as.Date(starting_date) + .x + n_length, length.out = n_length))
    dates <- date_list
    start_date <- map(date_list, ~ tail(.x, 1))
    mycal <- create.calendar(name="mycal", weekdays=days_off)
    end_day <- map(start_date, ~ add.bizdays(.x, future, cal=mycal))
    pred_dates <- map2(start_date, end_day, ~ tail(bizseq(.x, .y, cal=mycal), future))
    predictions <- map2(predictions, pred_dates, ~ as.data.frame(cbind(dates=.y, .x)))
    predictions <- map(predictions, ~ {.x$dates <- as.Date(.x$dates, origin = "1970-01-01"); return(.x)})
  }

  if(is.null(starting_date))
  {
    dates <- 1:n_length
    dates <- replicate(n_feat, dates, simplify = FALSE)
    pred_dates <- (n_length+1):(n_length+future)
    pred_dates <- replicate(n_feat, pred_dates, simplify = FALSE)
  }

  ###PREDICTION PLOT
  lower_name <- paste0("q", ((1-ci)/2) * 100)
  upper_name <- paste0("q", (ci+(1-ci)/2) * 100)

  plot <- pmap(list(data, predictions, targets, dates, pred_dates), ~ ts_graph(x_hist = ..4, y_hist = ..1, x_forcat = ..5, y_forcat = ..2[, "q50"],
                                                                               lower = ..2[, lower_name], upper = ..2[, upper_name], label_x = paste0("Extreme Gradient Boosting Machine Time Series Analysis (past = ", past ,", future = ", future,")"),
                                                                               label_y = paste0(str_to_title(..3), " Values"), dbreak = dbreak))

  toc(log = TRUE)
  time_log<-seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(training_error = training_error, testing_error = testing_error, predictions = predictions, pred_stats = pred_stats, plot = plot)
  return(outcome)
}



