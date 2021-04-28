#' audrex
#'
#' @param data A data frame with time series on columns and possibly a date column (not mandatory)
#' @param targets String. Names of ts features to be jointly analyzed: for each feature a distinct model is built using the others as regressors.
#' @param past Positive integer. The past dimension with number of time-steps in the past used for the prediction.
#' @param deriv Positive integer. Number of differentiation operations to perform on the original series. 0 = no change; 1: one diff; 2: two diff, and so on.
#' @param future Positive integer. The future dimension with number of time-steps to be predicted
#' @param shift Vector of positive integers. Allow for each ts feature to shift ahead of time. Zero means no shift. Length must be equal to the number of targets. Default: 0.
#' @param smoother Logical. Perform optimal smooting using standard loess. Default: FALSE
#' @param ci Confidence interval. Default: 0.8
#' @param holdout Positive numeric. Percentage of time series for holdout validation. Default: 0.5.
#' @param nrounds Positive numeric. Number of round for the extreme boosting machine. Look to xgboost for description. Default: 100.
#' @param patience Positive integer. Waiting rounds without improvement before xgboost stops. Default: 10
#' @param booster String. Optimization methods available are: "gbtree", "gblinear". Default: "gbtree".
#' @param max_depth Positive integer. Look to xgboost documentation for description. A vector with one or two positive integer for the search boundaries. The default value (NULL) sets automatic the values in c(1, 10).
#' @param eta Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatic the values in c(0.001, 1).
#' @param gamma Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatic the values in c(0.001, 100).
#' @param min_child_weight Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatic the values in c(1, 100).
#' @param subsample Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatic the values in c(0.1, 1).
#' @param colsample_bytree Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatic the values in c(0.1, 1).
#' @param lambda Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatic the values in c(0.1, 100).
#' @param alpha Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatic the values in c(0.1, 100).
#' @param verbose Logical. Default: TRUE
#' @param reg String. Learning objective function. Options are: "squarederror", "pseudohubererror".Default: "squarederror".
#' @param eval_metric String. Evaluation metric for the boosting algorithm. Options are: "rmse", "mae", "mape".Default: "mae".
#' @param starting_date Date. Initial date to assign temporal values to the series. Default: NULL (progressive numbers).
#' @param dbreak String. Minimum time marker for x-axis, in liberal form: i.e., "3 months", "1 week", "20 days". Default: NULL.
#' @param days_off String. Weekdays to exclude (i.e., c("saturday", "sunday")). Default: NULL.
#' @param min_set Positive integer. Minimun number for validation set in case of automatic resize of past dimension. Default: 30.
#' @param seed Random seed. Default: 42.
#' @param opt_metric String. Parameter for selecting the best model, averaging one-step error across all ts features. Default: "mae".
#' @param n_samp Positive integer. Number of samples for the Bayesian Optimization. Default: 15.
#' @param n_search Positive integer. Number of search steps for the Bayesian Optimization. Default: 15.
#' @param acq String. Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: "ucb".
#' @param kappa Positive numeric.Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: 2.576.
#' @param eps Positive numeric.Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: 0.
#' @param kernel List.Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: list(type = "exponential", power = 2).
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item best_par: the parameter of the best model selected through Bayesian Optimization
#' \item history: a table with the sampled models (n_samp + n_search), their parameters and optimization metric
#' \item best_model: results for the best selected model, including:
#' \itemize{
#' \item errors: training and testing errors for one-step and sequence for each ts feature (rmse, mae, mdae, mpe, mape, smape)
#' \item predictions: min, max, q25, q50, q75, quantiles at selected ci, mean, sd for each ts feature
#' \item pred_stats: some stats for each ts feature (iqr to min-max range, last-to-first iqr ratio, upside probability from ecdf)
#' }
#' \item time_log
#' }
#'
#' @export
#'
#' @import purrr
#' @import rBayesianOptimization
#' @importFrom fANCOVA loess.as
#' @importFrom imputeTS na_kalman
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period
#'
#'@examples
#'\donttest{
#'audrex(bitcoin_gold_oil, c("gold_close", "oil_Close"), past = 30, deriv = 1, future = 10)
#'}
#'
audrex <- function(data, targets, past, deriv, future, shift = 0, smoother = FALSE, ci = 0.8, holdout = 0.3, nrounds = 100, patience = 10,
                   booster = "gbtree", max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL, lambda = NULL, alpha = NULL,
                   verbose = FALSE, reg = "squarederror", eval_metric = "rmse", starting_date = NULL, dbreak = NULL, days_off = NULL, min_set = 30, seed = 42,
                   opt_metric = "mae", n_samp = 15, n_search = 15, acq = "ucb", kappa = 2.576, eps = 0, kernel = list(type = "exponential", power = 2))
{

  if(!(reg %in% c("squarederror", "pseudohubererror"))){stop("reg options are squarederror or pseudohubererror")}
  if(!(eval_metric %in% c("rmse", "mae", "mape"))){stop("eval_metric options are rmse, mae or mape")}

  if(booster=="gbtree")
  {
    if(is.null(max_depth)){max_depth <- c(1, 10)}
    if(is.null(eta)){eta <- c(0.001, 1)}
    if(is.null(gamma)){gamma <- c(0.001, 100)}
    if(is.null(min_child_weight)){min_child_weight <- c(1, 100)}
    if(is.null(subsample)){subsample <- c(0.1, 1)}
    if(is.null(colsample_bytree)){colsample_bytree <- c(0.1, 1)}

    body <- quote(
      {model <- xgb_forecast(data = data, targets, past, deriv, future, ci, holdout, nrounds, patience,
                             booster = "gbtree", max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda = NULL, alpha = NULL, verbose, reg, eval_metric,
                             starting_date, dbreak, days_off, include_training_error = FALSE, min_set)

      error <- mean(map_dbl(model$testing_error, ~ .x["one_step_error", opt_metric]), na.rm = TRUE)

      return(list(Score=-error, Pred=model))})

    hyper_params <- list(max_depth = as.integer(max_depth), min_child_weight = min_child_weight, eta = eta, gamma = gamma, subsample = subsample, colsample_bytree = colsample_bytree)
  }

  if(booster=="gblinear")
  {
    if(is.null(eta)){eta <- c(0.001, 1)}
    if(is.null(lambda)){lambda <- c(0.1, 100)}
    if(is.null(alpha)){alpha <- c(0.1, 100)}

    body <- quote(
      {model <- xgb_forecast(data = data, targets, past, deriv, future, ci, holdout, nrounds, patience,
                             booster = "gblinear", max_depth = NULL, eta, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL, lambda, alpha, verbose, reg, eval_metric,
                             starting_date, dbreak, days_off, include_training_error = FALSE, min_set)

      error <- mean(map_dbl(model$testing_error, ~ .x["one_step_error", opt_metric]), na.rm = TRUE)

      return(list(Score=-error, Pred=model))})

    hyper_params <- list(eta = eta, lambda = lambda, alpha = alpha)
  }



  #####

  tic.clearlog()
  tic("audrex")

  set.seed(seed)
  data <- data[, targets, drop = FALSE]
  if(anyNA(data)){data <- as.data.frame(purrr::map(data, ~ na_kalman(.x))); if(verbose == TRUE){message("kalman imputation on target and/or regressors\n")}}
  if(smoother==TRUE){data <- as.data.frame(map(data, ~ loess.as(x=1:n_length, y=.x)$fitted)); message("performing optimal smoothing\n")}
  n_feat <- ncol(data)

  ###SHIFT
  if(any(shift > 0) & length(shift)==n_feat)
  {
    data <- map2(data, shift, ~ tail(.x, n_length - .y))
    n_length <- min(map_dbl(data, ~ length(.x)))
    data <- map_df(data, ~ head(.x, n_length))
  }


  no_tuning <- map_lgl(hyper_params, ~ length(.x)== 1)
  tuned_hyper_param <- keep(hyper_params, !no_tuning)

  history <- NULL
  best_par <- NULL
  if(length(tuned_hyper_param)>0)
  {
  args <- make_alist(names(tuned_hyper_param))
  wrapper <- make_function(args, body)

  bop <- BayesianOptimization(wrapper, bounds = tuned_hyper_param, init_points = n_samp, n_iter = n_search, acq = acq, kappa = kappa, eps = eps, kernel = kernel, verbose = TRUE)

  var_names <- names(bop$Best_Par)
  best_par <- as.data.frame(matrix(bop$Best_Par, nrow=1))
  colnames(best_par) <- var_names

  best_par$test_metric <- bop$Best_Value * (-1)

  history <- as.data.frame(bop$History)
  colnames(history)[colnames(history)=="Value"] <- "test_metric"
  history$test_metric <- history$test_metric *(-1)
  }

  best_par <- round(unlist(c(hyper_params[no_tuning], best_par)), 3)

  best_model <- xgb_forecast(data = data, targets, past, deriv, future, ci, holdout, nrounds, patience,
                             booster, max_depth = best_par["max_depth"], eta = best_par["eta"], gamma = best_par["gamma"], min_child_weight = best_par["min_child_weight"],
                             subsample = best_par["subsample"], colsample_bytree = best_par["colsample_bytree"], lambda = best_par["lambda"], alpha = best_par["alpha"],
                             verbose, reg, eval_metric, starting_date, dbreak, days_off, include_training_error = TRUE, min_set)

  errors <- purrr::transpose(list(training_error = best_model$training_error, testing_error = best_model$testing_error))
  best_model$errors <- errors
  best_model <- list(errors = errors, predictions = best_model$predictions, pred_stats = best_model$pred_stats, plot = best_model$plot)

  toc(log = TRUE)
  time_log<-tail(seconds_to_period(round(parse_number(unlist(tic.log())), 0)), 1)

  outcome <- list(best_par = best_par, history = history, best_model = best_model, time_log = time_log)

  return(outcome)
}
