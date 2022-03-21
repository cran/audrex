#' audrex
#'
#' @param data A data frame with time features on columns.
#' @param n_sample Positive integer. Number of samples for the Bayesian Optimization. Default: 10.
#' @param n_search Positive integer. Number of search steps for the Bayesian Optimization. When the parameter is set to 0, optimization is shifted to Random Search. Default: 5,
#' @param smoother Logical. Perform optimal smoothing using standard loess. Default: FALSE
#' @param seq_len Positive integer. Number of time-steps to be predicted. Default: NULL (automatic selection)
#' @param diff_threshold Positive numeric. Minimum F-test threshold for differentiating each time feature (keep it low). Default: 0.001.
#' @param booster String. Optimization methods available are: "gbtree", "gblinear". Default: "gbtree".
#' @param norm Logical. Boolean flag to apply Yeo-Johson normalization. Default: NULL (automatic selection from random search or bayesian search).
#' @param n_dim Positive integer. Projection of time features in a lower dimensional space with n_dim features. The default value (NULL) sets automatically the values in c(1, n features).
#' @param ci Confidence interval. Default: 0.8.
#' @param min_set Positive integer. Minimun number for validation set in case of automatic resize of past dimension. Default: 30.
#' @param max_depth Positive integer. Look to xgboost documentation for description. A vector with one or two positive integer for the search boundaries. The default value (NULL) sets automatically the values in c(1, 8).
#' @param eta Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatically the values in c(0, 1).
#' @param gamma Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatically the values in c(0, 100).
#' @param min_child_weight Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatically the values in c(0, 100).
#' @param subsample Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatically the values in c(0, 1).
#' @param colsample_bytree Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric between (0, 1] for the search boundaries. The default value (NULL) sets automatically the values in c(0, 1).
#' @param lambda Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatically the values in c(0, 100).
#' @param alpha Positive numeric. Look to xgboost documentation for description. A vector with one or two positive numeric for the search boundaries. The default value (NULL) sets automatically the values in c(0, 100).
#' @param n_windows Positive integer. Number of (expanding) windows for cross-validation. Default: 3.
#' @param patience Positive numeric. Percentage of waiting rounds without improvement before xgboost stops. Default: 0.1
#' @param nrounds Positive numeric. Number of round for the extreme boosting machine. Look to xgboost for description. Default: 100.
#' @param dates Date. Vector of dates for the time series. Default: NULL (progressive numbers).
#' @param acq String. Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: "ucb".
#' @param kappa Positive numeric. Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: 2.576.
#' @param eps Positive numeric. Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: 0.
#' @param kernel List. Parameter for Bayesian Optimization. For reference see rBayesianOptimization documentation. Default: list(type = "exponential", power = 2).
#' @param seed Random seed. Default: 42.

#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item history: a table with the models from bayesian (n_sample + n_search) or random search (n_sample), their hyper-parameters and optimization metric, the weighted average rank
#' \item models: a list with the details for each model in history
#' \item best_model: results for the best selected model according to the weighted average rank, including:
#' \itemize{
#' \item predictions: min, max, q25, q50, q75, quantile at selected ci, mean, sd, skewness and kurtosis for each time feature
#' \item joint_error: max sequence error for the differentiated time features (max_rmse, max_mae, max_mdae, max_mape, max_mase, max_rae, max_rse, max_rrse, both for training and testing)
#' \item serie_errors: sequence error for the differentiated time features averaged across testing windows (rmse, mae, mdae, mape, mase, rae, rse, rrse, both for training and testing)
#' \item pred_stats: for each predicted time feature, IQR to range, divergence, risk ratio, upside probability, averaged across prediction time-points and at the terminal points
#' \item plots: a plot for each predicted time feature with highlighted median and confidence intervals
#' }
#' \item time_log
#' }
#'
#' @export
#'
#' @import purrr
#' @import tictoc
#' @importFrom fANCOVA loess.as
#' @importFrom imputeTS na_kalman
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period
#'
#'@examples
#'\donttest{
#'audrex(covid_in_europe[, 2:5], n_samp = 3, n_search = 2, seq_len = 10) ### BAYESIAN OPTIMIZATION
#'audrex(covid_in_europe[, 2:5], n_samp = 5, n_search = 0, seq_len = 10) ### RANDOM SEARCH WHEN n_search SET TO 0
#'}
#'
#'
audrex <- function(data, n_sample = 10, n_search = 5, smoother = FALSE, seq_len = NULL, diff_threshold = 0.001, booster = "gbtree", norm = NULL, n_dim = NULL, ci = 0.8, min_set = 30,
                   max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL, lambda = NULL, alpha = NULL,
                   n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL, acq = "ucb", kappa = 2.576, eps = 0, kernel = list(type = "exponential", power = 2), seed = 42)
{
  tic.clearlog()
  tic("time")

  if(anyNA(data)){data <- as.data.frame(map(data, ~ na_kalman(.x))); message("kalman imputation on target and/or regressors\n")}
  if(smoother==TRUE){data <- as.data.frame(map(data, ~ suppressWarnings(loess.as(x=1:nrow(data), y=.x)$fitted))); message("performing optimal smoothing\n")}

  deriv <- map_dbl(data, ~ best_deriv(.x, diff_threshold))

  if((n_sample >= 1 & n_search == 0)|(n_sample == 1 & n_search > 0)){search <- random_search(n_sample, data, booster, seq_len, deriv , norm, n_dim, ci, min_set,
                                                                                             max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha, n_windows, patience, nrounds, dates, seed)}

  if(n_sample > 1 & n_search >= 1){search <- bayesian_search(n_sample, n_search, booster, data, seq_len, deriv, norm, n_dim, ci, min_set,
                                                             max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha, n_windows, patience, nrounds, dates,
                                                             acq, kappa, eps, kernel)}

  history <- search$history
  models <- search$models
  best_idx <- which.min(history$wgt_avg_rank)

  best_model <- models[[best_idx]]

  toc(log = TRUE)
  time_log <- seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(history = history, models = models, best_model = best_model, time_log = time_log)

  return(outcome)
}
