#' support functions for audrex
#'
#' @param predictors A data frame with predictors on columns.
#' @param target A numeric vector with target variable.
#' @param booster String. Optimization methods available are: "gbtree", "gblinear". Default: "gbtree".
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
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @importFrom scales number
#' @importFrom stats lm quantile predict ecdf runif sd weighted.mean
#' @importFrom utils head tail
#' @importFrom modeest mlv1
#' @importFrom moments skewness kurtosis
#' @importFrom parallel detectCores
#' @importFrom narray split
#' @importFrom rBayesianOptimization BayesianOptimization
#' @import purrr
#' @import xgboost
#' @import ggplot2
#' @import stringr
#' @import Metrics


####
engine <- function(predictors, target, booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha, n_windows, patience, nrounds)
{
  nobs <- nrow(predictors)
  df <- data.frame(target, predictors)

  raw_collection <- list()
  test_collection <- list()

  w_index <- rep(1:(n_windows + 1), each = nobs/(n_windows + 1))
  w_index <- c(rep(1, nobs%%(n_windows + 1)), w_index)
  if(booster == "gbtree"){params <- list(booster = booster, max_depth = max_depth, eta = eta, gamma = gamma, min_child_weight = min_child_weight, subsample = subsample, colsample_bytree  = colsample_bytree, nthread = detectCores() - 1, objective = "reg:squarederror")}
  if(booster == "gblinear"){params <- list(booster = booster, eta = eta, lambda = lambda, alpha = alpha, nthread = detectCores() - 1, objective = "reg:squarederror")}

  for(w in 1:n_windows)
  {
    train_set <- df[w_index <= w,, drop = FALSE]
    test_set <- df[w_index == w + 1,, drop = FALSE]
    dtrain <- xgb.DMatrix(as.matrix(train_set[, -1]), label = as.matrix(train_set[, 1]))
    dtest <- xgb.DMatrix(as.matrix(test_set[, -1]), label = as.matrix(test_set[, 1]))
    watchlist <- list(train = dtrain, eval = dtest)

    if(is.numeric(patience)){esr <- patience * nrounds}
    if(is.integer(patience)){esr <- patience}

    model <- xgb.train(params, dtrain, nrounds = nrounds, watchlist, verbose = 0, early_stopping_rounds = patience * 1000)

    train_predictions <- predict(model, as.matrix(train_set[,-1]))
    fix_index <- (is.finite(train_predictions) & train_predictions != 0) & (is.finite(train_set[,1]) & train_set[,1] != 0)
    training_errors <- map_dbl(list(rmse, mae, mdae, mape, mase, rae, rse, rrse), ~ .x(train_set[,1][fix_index], train_predictions[fix_index]))
    test_predictions <- predict(model, as.matrix(test_set[,-1]))
    fix_index <- (is.finite(test_predictions) & test_predictions != 0) & (is.finite(test_set[,1]) & test_set[,1] != 0)
    testing_errors <- map_dbl(list(rmse, mae, mdae, mape, mase, rae, rse, rrse), ~ .x(test_set[,1][fix_index], test_predictions[fix_index]))

    raw_collection[[w]] <- test_set[,1] - test_predictions

    errors <- rbind(training_errors, testing_errors)
    rownames(errors) <- c("training", "testing")
    colnames(errors) <- c("rmse", "mae", "mdae", "mape", "mase", "rae", "rse", "rrse")

    test_collection[[w]] <- errors
  }

  raw_error <- unlist(raw_collection)
  errors <- Reduce("+", test_collection)/n_windows
  model <- xgboost(data = as.matrix(df[,-1]), label = as.matrix(df[,1]), params = params, nrounds = nrounds, early_stopping_rounds = patience * 1000, verbose = 0)

  outcome <- list(model = model, errors = errors, raw_error = raw_error)

  return(outcome)
}

##########
sequencer <- function(seq_len, ts_set, target, deriv, ci = 0.8, min_set = 30, booster = "gbtree",
                      max_depth = 4, eta = 1, gamma = 1, min_child_weight = 1, subsample = 0.5, colsample_bytree = 0.7, lambda = 1, alpha = 1,
                      n_windows = 3, patience = 0.1, nrounds = 100, feat_name, dates = NULL)
{
  all_positive_check <- all(target >= 0)
  all_negative_check <- all(target <= 0)

  diff_model <- recursive_diff(target, deriv)
  target <- diff_model$vector
  ts_set <- smart_tail(ts_set, - deriv)

  sequence <- map(1:seq_len, ~ engine(head(ts_set, -.x), tail(target, - .x), booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha, n_windows, patience, nrounds))
  models <- map(sequence, ~ .x$model)
  raw_errors <- map(sequence, ~ .x$raw_error)
  seq_errors <- Reduce("+", map(sequence, ~ .x$errors))/seq_len
  sequence <- map_dbl(models, ~ predict(.x, as.matrix(tail(ts_set, 1))))
  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))
  error_integration <- as.data.frame(map2(sequence, raw_errors, ~ .x + sample(.y, size = 1000, replace = TRUE, prob = rep(1:n_windows, each = length(.y)/n_windows))))####SAMPLING BY WINDOWS WEIGHTS???
  if(seq_len > 1){integrated <- t(apply(error_integration, 1, function(x) invdiff(x, diff_model$tail_value)))}
  if(seq_len == 1){integrated <- matrix(apply(error_integration, 1, function(x) invdiff(x, diff_model$tail_value)), ncol = 1)}
  if(all_positive_check){integrated[integrated < 0] <- 0}
  if(all_negative_check){integrated[integrated > 0] <- 0}
  qfun <- function(x) {c(min = min(x, na.rm = TRUE), quantile(x, quants, na.rm = TRUE), max = max(x, na.rm = TRUE), mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE), mode = suppressWarnings(mlv1(x, method = "shorth", na.rm = TRUE)), skewness = skewness(x, na.rm=TRUE), kurtosis = kurtosis(x, na.rm=TRUE))}
  predicted <- t(round(apply(integrated, 2, qfun), 3))
  if(is.null(dim(predicted))){predicted <- as.data.frame(matrix(predicted, 1)); colnames(predicted) <- c("min", paste0(quants * 100, "%"), "max", "mean", "sd", "mode", "skewness", "kurtosis")}
  rownames(predicted) <- NULL

  avg_iqr_to_range <- round(mean((predicted[,"75%"] - predicted[,"25%"])/(predicted[,"max"] - predicted[,"min"])), 3)
  last_to_first_iqr <- round((predicted[seq_len,"75%"] - predicted[seq_len,"25%"])/(predicted[1,"75%"] - predicted[1,"25%"]), 3)
  iqr_stats <- c(avg_iqr_to_range, last_to_first_iqr)
  names(iqr_stats) <- c("avg_iqr_to_range", "terminal_iqr_ratio")

  avg_risk_ratio <- round(mean((predicted[,"max"] - predicted[,"50%"])/(predicted[,"50%"] - predicted[,"min"])), 3)
  last_to_risk_ratio <- round(((predicted[seq_len,"max"] - predicted[seq_len,"50%"])/(predicted[seq_len,"50%"] - predicted[seq_len,"min"]))/((predicted[1,"max"] - predicted[1,"50%"])/(predicted[1,"50%"] - predicted[1,"min"])), 3)
  risk_stats <- c(avg_risk_ratio, last_to_risk_ratio)
  names(risk_stats) <- c("avg_risk_ratio", "terminal_risk_ratio")

  div_stats <- sequential_divergence(integrated)
  upp_stats <- upside_probability(integrated)

  pred_stats <- round(c(iqr_stats, risk_stats, div_stats, upp_stats), 3)

  n_obs <- nrow(ts_set)
  target <- invdiff(target, diff_model$head_value)
  if(is.null(dates)){hist_dates <- 1:length(target); forcat_dates <- (n_obs + 1):(n_obs + seq_len)}
  if(!is.null(dates)){hist_dates <- tail(dates, length(target)); forcat_dates <- seq.Date(tail(dates, 1), tail(dates, 1) + seq_len * mean(diff(dates)), length.out = seq_len)}
  x_lab <- paste0("Forecasting Horizon for sequence n = ", seq_len)
  y_lab <- paste0("Forecasting Values for ", feat_name)

  plot <- ts_graph(x_hist = hist_dates, y_hist = target, x_forcat = forcat_dates, y_forcat = predicted[, 4], lower = predicted[, 2], upper = predicted[, 6], forcat_band = "deepskyblue", forcat_line = "deepskyblue4",
                   label_x = x_lab, label_y = y_lab)

  outcome <- list(models = models, predicted = predicted, seq_errors = seq_errors, pred_stats = pred_stats, plot = plot)
  return(outcome)
}

#######
hood <- function(ts_set, seq_len, deriv, norm = TRUE, n_dim, ci = 0.8, min_set = 30, booster = "gbtree",
                 max_depth = 4, eta = 1, gamma = 1, min_child_weight = 1, subsample = 0.5,
                 colsample_bytree = 0.7, lambda = 1, alpha = 1, n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL)
{
  targets <- split(ts_set, along = 2)
  feat_names <- colnames(ts_set)

  if(norm == TRUE){
    ###minmax <- function(x){(x - min(x))/(diff(range(x)))}
    ts_set <- as.data.frame(apply(ts_set, 2, function(x) optimized_yjt(x)$transformed))###NORMALIZATION
  }

  n_feats <- ncol(ts_set)

  if(n_feats > 1 && n_dim < n_feats){
    svd_model <- svd(ts_set)
    ts_set <- as.data.frame(svd_model$u[, 1:n_dim] %*% diag(svd_model$d[1:n_dim], n_dim, n_dim))
    colnames(ts_set) <- paste0("dim", 1:n_dim)}

  models <- pmap(list(targets, deriv, feat_names), ~ sequencer(seq_len, ts_set, ..1, ..2, ci, min_set, booster, max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha, n_windows, patience, nrounds, ..3, dates))
  serie_errors <- map(models, ~ round(.x$seq_errors, 3))
  max_train_error <- apply(Reduce(rbind, map(serie_errors, ~ .x[1,])), 2, max)
  max_test_error <- apply(Reduce(rbind, map(serie_errors, ~ .x[2,])), 2, max)
  joint_error <- round(rbind(max_train_error, max_test_error), 3)
  rownames(joint_error) <- c("train", "test")
  colnames(joint_error) <- paste0("max_", c("rmse", "mae", "mdae", "mape", "mase", "rae", "rse", "rrse"))
  predictions <- map(models, ~ .x$predicted)

  if(is.null(dates)){predictions <- map(predictions, ~ {rownames(.x) <- paste0("t", 1:seq_len); return(.x)})}
  if(!is.null(dates)){
    predicted_dates <- seq.Date(tail(dates, 1), tail(dates, 1) + seq_len * mean(diff(dates)), length.out = seq_len)
    predictions <- map(predictions, ~ {.x <- as.data.frame(.x); rownames(.x) <- predicted_dates; return(.x)})}

  plots <- map(models, ~ .x$plot)
  pred_stats <- as.data.frame(map(models, ~ .x$pred_stats))

  outcome <- list(predictions = predictions, joint_error = joint_error, serie_errors = serie_errors, pred_stats = pred_stats, plots = plots)

  return(outcome)
}




##############
ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "darkorange", forcat_line = "darkorange", hist_line = "gray43", label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{
  all_data <- data.frame(x_all = c(x_hist, x_forcat), y_all = c(y_hist, y_forcat))
  forcat_data <- data.frame(x_forcat = x_forcat, y_forcat = y_forcat)

  if(!is.null(lower) & !is.null(upper)){forcat_data$lower <- lower; forcat_data$upper <- upper}

  plot <- ggplot()+geom_line(data = all_data, aes_string(x = "x_all", y = "y_all"), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes(x = x_forcat, ymin = lower, ymax = upper), alpha = 3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes(x = x_forcat, y = y_forcat), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}

recursive_diff <- function(vector, deriv)
{
  vector <- unlist(vector)
  head_value <- vector("numeric", deriv)
  tail_value <- vector("numeric", deriv)
  if(deriv==0){head_value = NULL; tail_value = NULL}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector = vector, head_value = head_value, tail_value = tail_value)
  return(outcome)
}

invdiff <- function(vector, heads, add = FALSE)
{
  vector <- unlist(vector)
  if(is.null(heads)){return(vector)}
  for(d in length(heads):1){vector <- cumsum(c(heads[d], vector))}
  if(add == FALSE){return(vector[-c(1:length(heads))])} else {return(vector)}
}


smart_head <- function(x, n)
{
  if(n != 0){return(head(x, n))}
  if(n == 0){return(x)}
}

smart_tail <- function(x, n)
{
  if(n != 0){return(tail(x, n))}
  if(n == 0){return(x)}
}


###
upside_probability <- function(m)
{
  matrix <- t(as.matrix(m))
  n <- nrow(matrix)
  if(n == 1){return(c("avg_upside_prob" = NA, "terminal_upside_prob" = NA))}
  growths <- matrix[-1,]/matrix[-n,] - 1
  if(is.matrix(growths)){avg_upp <- round(mean(apply(growths, 1, function(x) mean(x[is.finite(x)] > 0, na.omit = TRUE)), na.omit = TRUE), 3)}
  if(!is.matrix(growths)){avg_upp <- round(mean(growths[is.finite(growths)] > 0, na.omit = TRUE), 3)}
  terminal_growth <- matrix[n,]/matrix[1,] - 1
  last_to_first_upp <- round(mean(terminal_growth[is.finite(terminal_growth)] > 0, na.omit = TRUE), 3)
  upp_stats <- c(avg_upp, last_to_first_upp)
  names(upp_stats) <- c("avg_upside_prob", "terminal_upside_prob")
  return(upp_stats)
}


##########
sequential_divergence <- function(m)
{
  matrix <- t(as.matrix(m))
  n <- nrow(matrix)
  s <- seq(min(matrix), max(matrix), length.out = 100)
  if(n == 1){return(c("max_divergence" = NA, "terminal_divergence" = NA))}
  dens <- apply(matrix, 1, ecdf, simplify = FALSE)
  backward <- dens[-n]
  forward <- dens[-1]
  seq_div <- map2_dbl(forward, backward, ~ abs(max(.x(s) - .y(s))))
  avg_seq_div <- round(mean(seq_div), 3)
  end_to_end_div <- abs(max(dens[[n]](s) - dens[[1]](s)))
  div_stats <- c(avg_seq_div, end_to_end_div)
  names(div_stats) <- c("max_divergence", "terminal_divergence")
  return(div_stats)
}

###

bayesian_search <- function(n_sample = 10, n_search = 5, booster, data, seq_len = NULL , deriv, norm = NULL, n_dim = NULL, ci = 0.8, min_set = 30,
                            max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL, lambda = NULL, alpha = NULL,
                            n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL, acq = "ucb", kappa = 2.576, eps = 0, kernel = list(type = "exponential", power = 2))
{

  n_obs <- nrow(data)
  n_feats <- ncol(data)

  if(booster == "gbtree"){

    sl_boundaries <- NULL
    norm_boundaries <- NULL
    dim_boundaries <- NULL
    depth_boundaries <- NULL
    eta_boundaries <- NULL
    gamma_boundaries <- NULL
    mcw_boundaries <- NULL
    ss_boundaries <- NULL
    csbt_boundaries <- NULL
    tuning_index <- rep(TRUE, 9)

    if(is.null(seq_len)){sl_boundaries <- c(max(deriv)+1, n_obs/3)} else {ifelse(length(seq_len) > 1, sl_boundaries <- seq_len, tuning_index[1] <- FALSE)}
    if(is.null(norm)){norm_boundaries <- c(FALSE, TRUE)} else {tuning_index[2] <- FALSE}
    if(is.null(n_dim)){dim_boundaries <- c(1, n_feats)} else {ifelse(length(n_dim) > 1, dim_boundaries <- n_dim, tuning_index[3] <- FALSE)}
    if(is.null(max_depth)){depth_boundaries <- c(1, 8)} else {ifelse(length(max_depth) > 1, depth_boundaries <- max_depth, tuning_index[4] <- FALSE)}
    if(is.null(eta)){eta_boundaries <- c(0, 1)} else {ifelse(length(eta) > 1, eta_boundaries <- eta, tuning_index[5] <- FALSE)}
    if(is.null(gamma)){gamma_boundaries <- c(0, 100)} else {ifelse(length(gamma) > 1, gamma_boundaries <- gamma, tuning_index[6] <- FALSE)}
    if(is.null(min_child_weight)){mcw_boundaries <- c(0, 100)} else {ifelse(length(min_child_weight) > 1, mcw_boundaries <- min_child_weight, tuning_index[7] <- FALSE)}
    if(is.null(subsample)){ss_boundaries <- c(0, 1)} else {ifelse(length(subsample) > 1, ss_boundaries <- subsample, tuning_index[8] <- FALSE)}
    if(is.null(colsample_bytree)){csbt_boundaries <- c(0, 1)} else {ifelse(length(colsample_bytree) > 1, csbt_boundaries <- colsample_bytree, tuning_index[9] <- FALSE)}
    tuned_params <- list(seq_len = as.integer(sl_boundaries), norm = as.integer(norm_boundaries), n_dim = as.integer(dim_boundaries), max_depth = as.integer(depth_boundaries), eta = eta_boundaries, gamma = gamma_boundaries, min_child_weight = mcw_boundaries, subsample = ss_boundaries, colsample_bytree = csbt_boundaries)[tuning_index]
  }


  if(booster == "gblinear"){

    sl_boundaries <- NULL
    norm_boundaries <- NULL
    dim_boundaries <- NULL
    eta_boundaries <- NULL
    lambda_boundaries <- NULL
    alpha_boundaries <- NULL
    tuning_index <- rep(TRUE, 6)

    if(is.null(seq_len)){sl_boundaries <- c(max(deriv)+1, n_obs/3)} else {ifelse(length(seq_len) > 1, sl_boundaries <- seq_len, tuning_index[1] <- FALSE)}
    if(is.null(norm)){norm_boundaries <- c(FALSE, TRUE)} else {tuning_index[2] <- FALSE}
    if(is.null(n_dim)){dim_boundaries <- c(1, n_feats)} else {ifelse(length(n_dim) > 1, dim_boundaries <- n_dim, tuning_index[3] <- FALSE)}
    if(is.null(eta)){eta_boundaries <- c(0, 1)} else {ifelse(length(eta) > 1, eta_boundaries <- eta, tuning_index[4] <- FALSE)}
    if(is.null(lambda)){lambda_boundaries <- c(0, 100)} else {ifelse(length(lambda) > 1, lambda_boundaries <- lambda, tuning_index[5] <- FALSE)}
    if(is.null(alpha)){alpha_boundaries <- c(0, 100)} else {ifelse(length(alpha) > 1, alpha_boundaries <- alpha, tuning_index[6] <- FALSE)}
    tuned_params <- list(seq_len = as.integer(sl_boundaries), norm = as.integer(norm_boundaries), n_dim = as.integer(dim_boundaries), eta = eta_boundaries, lambda = lambda_boundaries, alpha = alpha_boundaries)[tuning_index]
  }


  make_alist <- function(args) {
    res <- replicate(length(args), substitute())
    names(res) <- args
    res}

  make_function <- function(args, body, env = parent.frame()) {
    args <- as.pairlist(args)
    eval(call("function", args, body), env)}

  if(booster == "gbtree"){
    body <- quote(
      {model <- hood(data, seq_len , deriv, norm, n_dim, ci, min_set, booster = "gbtree",
                     max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree,
                     lambda = NULL, alpha = NULL, n_windows, patience, nrounds, dates)

      error <- mean(model$joint_error[2, c("max_rmse", "max_mae", "max_mdae")])

      return(list(Score = -error, Pred = model))})}

  if(booster == "gblinear"){
    body <- quote(
      {model <- hood(data, seq_len, deriv, norm, n_dim, ci, min_set, booster = "gblinear",
                     max_depth = NULL, eta, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL,
                     lambda, alpha, n_windows, patience, nrounds, dates)

      error <- mean(model$joint_error[2, 1:3])

      return(list(Score = -error, Pred = model))})}

  args <- make_alist(names(tuned_params))
  wrapper <- make_function(args, body)

  bop <- BayesianOptimization(wrapper, bounds = tuned_params, init_points = n_sample, n_iter = n_search, acq = acq, kappa = kappa, eps = eps, kernel = kernel, verbose = FALSE)

  errors <- round(Reduce(rbind, map(bop$Pred, ~ .x[[2]][2,])), 3)
  weights <- apply(errors, 2, function(x) {abs(sd(x[is.finite(x)], na.rm = TRUE)/mean(x[is.finite(x)], na.rm = TRUE))})
  finite_w <- is.finite(weights)
  wgt_avg_rank <- round(apply(apply(abs(errors[, finite_w, drop = FALSE]), 2, rank), 1, weighted.mean, w = weights[finite_w]), 2)

  history <- round(as.data.frame(bop$History), 3)
  history <- cbind(history[, - c(1, ncol(history))], errors, wgt_avg_rank)
  history$norm <- as.logical( history$norm)
  rownames(history) <- NULL
  models <- as.list(bop$Pred)
  models <- map(models, ~ {names(.x) <- c("predictions", "joint_error", "serie_errors", "pred_stats", "plots"); return(.x)})

  outcome <- list(history = history, models = models)
  return(outcome)

}

#########
random_search <- function(n_sample, data, booster = "gbtree", seq_len = NULL, deriv, norm = NULL, n_dim = NULL, ci = 0.8, min_set = 30,
                          max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL, subsample = NULL,
                          colsample_bytree = NULL, lambda = NULL, alpha = NULL, n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL, seed = 42)
{
  set.seed(seed)
  n_obs <- nrow(data)
  n_feats <- ncol(data)

  if(is.null(seq_len)){sl_set <- sample(n_obs/3, n_sample, replace = TRUE)} else {ifelse(length(seq_len) > 1, sl_set <- sample(seq_len, n_sample, replace = TRUE), sl_set <- rep(seq_len, n_sample))}
  sl_set[sl_set <= max(deriv)] <- max(deriv) + 1
  if(is.null(norm)){norm_set <- sample(c(TRUE, FALSE), n_sample, replace = TRUE)} else {norm_set <- rep(norm, n_sample)}
  if(is.null(n_dim)){dim_set <- sample(n_feats, n_sample, replace = TRUE)} else {ifelse(length(n_dim) > 1, dim_set <- sample(n_dim, n_sample, replace = TRUE), dim_set <- rep(n_dim, n_sample))}
  dim_set[dim_set > n_feats] <- n_feats

  if(booster == "gbtree"){
    if(is.null(max_depth)){depth_set <- sample(8, n_sample, replace = TRUE)} else {ifelse(length(max_depth) > 1, depth_set <- sample(max_depth, n_sample, replace = TRUE), depth_set <- rep(max_depth, n_sample))}
    if(is.null(eta)){eta_set <- runif(n_sample)} else {ifelse(length(eta) > 1, eta_set <- sample(eta, n_sample, replace = TRUE), eta_set <- rep(eta, n_sample))}
    if(is.null(gamma)){gamma_set <- runif(n_sample, 0, 100)} else {ifelse(length(gamma) > 1, gamma_set <- sample(gamma, n_sample, replace = TRUE), gamma_set <- rep(gamma, n_sample))}
    if(is.null(min_child_weight)){mcw_set <- sample(100, n_sample, replace = TRUE)} else {ifelse(length(min_child_weight) > 1, mcw_set <- sample(min_child_weight, n_sample, replace = TRUE), mcw_set <- rep(min_child_weight, n_sample))}
    if(is.null(subsample)){ss_set <- runif(n_sample)} else {ifelse(length(subsample) > 1, ss_set <- sample(subsample, n_sample, replace = TRUE), ss_set <- rep(subsample, n_sample))}
    if(is.null(colsample_bytree)){csbt_set <- runif(n_sample)} else {ifelse(length(colsample_bytree) > 1, csbt_set <- sample(colsample_bytree, n_sample, replace = TRUE), csbt_set <- rep(colsample_bytree, n_sample))}
    hyper_params <- list(sl_set, norm_set, dim_set, depth_set, eta_set, gamma_set, mcw_set, ss_set, csbt_set)

    exploration <- pmap(hyper_params, ~ tryCatch(hood(data, seq_len = ..1, deriv, norm = ..2, n_dim = ..3, ci, min_set, booster,
                                                             max_depth = ..4, eta = ..5, gamma = ..6, min_child_weight = ..7, subsample = ..8,
                                                             colsample_bytree = ..9, lambda = NULL, alpha = NULL, n_windows, patience, nrounds, dates), error = function(e) NA))
  }

  if(booster == "gblinear"){
    if(is.null(eta)){eta_set <- runif(n_sample)} else {ifelse(length(eta) > 1, eta_set <- sample(eta, n_sample, replace = TRUE), eta_set <- rep(eta, n_sample))}
    if(is.null(lambda)){lambda_set <- runif(n_sample, 0, 100)} else {ifelse(length(lambda) > 1, lambda_set <- sample(lambda, n_sample, replace = TRUE), lambda_set <- rep(lambda, n_sample))}
    if(is.null(alpha)){alpha_set <- runif(n_sample, 0, 100)} else {ifelse(length(alpha) > 1, alpha_set <- sample(alpha, n_sample, replace = TRUE), alpha_set <- rep(alpha, n_sample))}
    hyper_params <- list(sl_set, norm_set, dim_set, eta_set, lambda_set, alpha_set)

    exploration <- pmap(hyper_params, ~ tryCatch(hood(data, seq_len = ..1, deriv, norm = ..2, n_dim = ..3, ci, min_set, booster,
                                                             max_depth = NULL, eta = ..4, gamma = NULL, min_child_weight = NULL, subsample = NULL,
                                                             colsample_bytree = NULL, lambda = ..5, alpha = ..6, n_windows, patience, nrounds, dates), error = function(e) NA))
  }

  not_na <- !is.na(exploration)
  exploration <- exploration[not_na]

  if(n_sample == 1){
    history <- t(Reduce(rbind, map(exploration, ~ .x$joint_error[2,])))
    wgt_avg_rank <- 1}

  if(n_sample > 1){
    history <- Reduce(rbind, map(exploration, ~ .x$joint_error[2,]))
    rownames(history) <- NULL
    weights <- apply(history, 2, function(x) {abs(sd(x[is.finite(x)], na.rm = TRUE)/mean(x[is.finite(x)], na.rm = TRUE))})
    finite_w <- is.finite(weights)
    wgt_avg_rank <- round(apply(apply(abs(history[, finite_w, drop = FALSE]), 2, rank), 1, weighted.mean, w = weights[finite_w]), 2)}

  if(booster == "gbtree"){history <- cbind(data.frame(seq_len = sl_set[not_na], norm = norm_set[not_na], n_dim = dim_set[not_na], max_depth = depth_set[not_na], eta = round(eta_set[not_na], 2), gamma = round(gamma_set[not_na], 2), min_child_weight = mcw_set[not_na], subsample = round(ss_set[not_na], 4), colsample_bytree = round(csbt_set[not_na], 4)), round(history, 3), wgt_avg_rank)}
  if(booster == "gblinear"){history <- cbind(data.frame(seq_len = sl_set[not_na], norm = norm_set[not_na], n_dim = dim_set[not_na], eta = round(eta_set[not_na], 2), lambda = round(lambda_set[not_na], 2), alpha = round(alpha_set[not_na], 2)), round(history, 3), wgt_avg_rank)}

  outcome <- list(history = history, models = exploration)
}


optimized_yjt <- function(vector, precision = 100)
{

  yjt_fun <- function(x, lambda = 0.5)
  {
    yjt <- vector(mode = "numeric", length = length(x))

    for(i in 1:length(x))
    {
      if(x[i] >= 0 & lambda != 0){yjt[i] <- ((x[i]+1)^lambda - 1)/lambda}
      if(x[i] >= 0 & lambda == 0){yjt[i] <- log(x[i]+1)}
      if(x[i] < 0 & lambda != 2){yjt[i] <- -((-x[i]+1)^(2 - lambda) - 1)/(2 - lambda)}
      if(x[i] < 0 & lambda == 2){yjt[i] <- -log(-x[i]+1)}
    }
    return(yjt)
  }


  inv_yjt <- function(x, lambda = 0.5)
  {
    inv_yjt <- vector(mode = "numeric", length = length(x))

    for(i in 1:length(x))
    {
      if(x[i] >= 0 & lambda != 0){inv_yjt[i] <- exp(log(x[i] * lambda + 1)/lambda) - 1}
      if(x[i] >= 0 & lambda == 0){inv_yjt[i] <- exp(x[i]) - 1}
      if(x[i] < 0 & lambda != 2){inv_yjt[i] <- 1 - exp(log(1 - x[i]*(2 - lambda))/(2 - lambda))}
      if(x[i] < 0 & lambda == 2){inv_yjt[i] <- 1 - exp(-x[i])}
    }

    return(inv_yjt)
  }

  lambda_seq <- seq(0, 2, length.out = precision)
  cor_seq <- map_dbl(lambda_seq, ~ cor(yjt_fun(vector, .x), vector))
  best_lambda <- lambda_seq[which.max(cor_seq)]

  transformed <- yjt_fun(vector, best_lambda)
  direct_fun <- function(x){yjt_fun(x, best_lambda)}
  inverse_fun <- function(x){inv_yjt(x, best_lambda)}

  outcome <- list(transformed = transformed, best_lambda = best_lambda, direct_fun = direct_fun, inverse_fun = inverse_fun)
  return(outcome)
}


best_deriv <- function(ts, max_diff = 3, thresh = 0.001)
{
  pvalues <- vector(mode = "double", length = as.integer(max_diff))

  for(d in 1:(max_diff + 1))
  {
    model <- lm(ts ~ t, data.frame(ts, t = 1:length(ts)))
    pvalues[d] <- with(summary(model), pf(fstatistic[1], fstatistic[2], fstatistic[3],lower.tail=FALSE))
    ts <- diff(ts)
  }

  best <- tail(cumsum(pvalues < thresh), 1)

  return(best)
}
