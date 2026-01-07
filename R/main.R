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
#' @import xgboost
#' @importFrom imputeTS na_kalman
#' @importFrom stats runif weighted.mean
#' @importFrom utils tail
#' @import grDevices
#' @import graphics
#'
#'
#'@examples
#'\donttest{
#'audrex(covid_in_europe[, 2:5], n_samp = 3, n_search = 2, seq_len = 10) ### BAYESIAN OPTIMIZATION
#'audrex(covid_in_europe[, 2:5], n_samp = 5, n_search = 0, seq_len = 10) ### RANDOM SEARCH
#'}
#'
#'
audrex <- function(
    data,
    n_sample = 10,
    n_search = 5,
    smoother = FALSE,
    seq_len = NULL,
    diff_threshold = 0.001,
    booster = "gbtree",
    norm = NULL,
    n_dim = NULL,
    ci = 0.8,
    min_set = 30,
    max_depth = NULL,
    eta = NULL,
    gamma = NULL,
    min_child_weight = NULL,
    subsample = NULL,
    colsample_bytree = NULL,
    lambda = NULL,
    alpha = NULL,
    n_windows = 3,
    patience = 0.1,
    nrounds = 100,
    dates = NULL,
    acq = "ucb",
    kappa = 2.576,
    eps = 0,
    kernel = list(type = "exponential", power = 2),
    seed = 42
) {
  t0 <- proc.time()[["elapsed"]]

  if (!is.data.frame(data)) data <- as.data.frame(data)
  if (ncol(data) < 1) stop("`data` must have at least 1 column.")
  if (ncol(data) < 2) {
    n_dim <- 1L
    norm <- FALSE
  }

  if (anyNA(data)) {
    data <- as.data.frame(lapply(data, function(x) imputeTS::na_kalman(x)))
    message("kalman imputation on target and/or regressors\n")
  }

  if (isTRUE(smoother)) {
    data <- as.data.frame(lapply(data, .optimal_loess_smooth))
    message("performing optimal smoothing (loess)\n")
  }

  deriv <- vapply(data, function(x) best_deriv(x, max_diff = 3, thresh = diff_threshold), numeric(1))

  if ((n_sample >= 1 && n_search == 0) || (n_sample == 1 && n_search > 0)) {
    search <- random_search(
      n_sample = n_sample, data = data, booster = booster, seq_len = seq_len, deriv = deriv,
      norm = norm, n_dim = n_dim, ci = ci, min_set = min_set,
      max_depth = max_depth, eta = eta, gamma = gamma, min_child_weight = min_child_weight,
      subsample = subsample, colsample_bytree = colsample_bytree, lambda = lambda, alpha = alpha,
      n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates, seed = seed
    )
  } else if (n_sample > 1 && n_search >= 1) {
    search <- bayesian_search(
      n_sample = n_sample, n_search = n_search, booster = booster, data = data,
      seq_len = seq_len, deriv = deriv, norm = norm, n_dim = n_dim, ci = ci, min_set = min_set,
      max_depth = max_depth, eta = eta, gamma = gamma, min_child_weight = min_child_weight,
      subsample = subsample, colsample_bytree = colsample_bytree, lambda = lambda, alpha = alpha,
      n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates,
      acq = acq, kappa = kappa, eps = eps, kernel = kernel
    )
  } else {
    stop("Invalid configuration: need n_sample >= 1; if n_search>0 then n_sample must be >1.")
  }

  history <- search$history
  models <- search$models
  best_idx <- which.min(history$wgt_avg_rank)
  best_model <- models[[best_idx]]

  elapsed <- proc.time()[["elapsed"]] - t0
  time_log <- .format_seconds(elapsed)

  list(history = history, models = models, best_model = best_model, time_log = time_log)
}

#' @keywords internal
# -----------------------------
# engine (xgboost training + expanding-window CV)
# -----------------------------

engine <- function(
    predictors, target, booster,
    max_depth, eta, gamma, min_child_weight, subsample, colsample_bytree, lambda, alpha,
    n_windows, patience, nrounds
) {
  if (!is.data.frame(predictors)) predictors <- as.data.frame(predictors)
  nobs <- nrow(predictors)
  if (length(target) != nobs) stop("engine: predictors and target length mismatch.")

  # Hard guard: ensure folds feasible
  min_train_for_folds <- 2L * (as.integer(n_windows) + 1L)
  if (nobs < min_train_for_folds) {
    stop("engine: too few rows (", nobs, ") for n_windows=", n_windows,
         ". Need at least ", min_train_for_folds, " rows.")
  }

  folds <- .make_time_folds(nobs, n_windows = n_windows)
  nthread <- .safe_nthread()

  if (booster == "gbtree") {
    params <- list(
      booster = "gbtree",
      objective = "reg:squarederror",
      eval_metric = "rmse",
      max_depth = as.integer(max_depth),
      eta = eta,
      gamma = gamma,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      nthread = nthread
    )
  } else if (booster == "gblinear") {
    params <- list(
      booster = "gblinear",
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = eta,
      lambda = lambda,
      alpha = alpha,
      nthread = nthread
    )
  } else stop("Unknown booster: ", booster)

  esr <- .as_early_stopping_rounds(patience, nrounds)

  xmat <- as.matrix(predictors)
  yvec <- as.numeric(target)

  raw_residuals <- vector("list", n_windows)
  err_list <- vector("list", n_windows)

  for (w in seq_len(n_windows)) {
    tr <- folds[[w]]$train
    te <- folds[[w]]$test

    dtrain <- xgboost::xgb.DMatrix(data = xmat[tr, , drop = FALSE], label = yvec[tr])
    dtest  <- xgboost::xgb.DMatrix(data = xmat[te, , drop = FALSE], label = yvec[te])

    # NEW API: use evals (not watchlist)
    evals <- list(train = dtrain, eval = dtest)

    mdl <- suppressWarnings(
      xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = as.integer(nrounds),
        evals = evals,
        verbose = 0,
        early_stopping_rounds = as.integer(esr)
      )
    )

    # NEW API: xgboost::predict is not exported; use S3 method via base predict()
    pred_tr <- suppressWarnings(as.numeric(stats::predict(mdl, dtrain)))
    pred_te <- suppressWarnings(as.numeric(stats::predict(mdl, dtest)))

    fix_tr <- is.finite(pred_tr) & is.finite(yvec[tr])
    fix_te <- is.finite(pred_te) & is.finite(yvec[te])

    train_errors <- .reg_errors(yvec[tr][fix_tr], pred_tr[fix_tr])
    test_errors  <- .reg_errors(yvec[te][fix_te], pred_te[fix_te])

    raw_residuals[[w]] <- yvec[te][fix_te] - pred_te[fix_te]
    err_list[[w]] <- rbind(training = train_errors, testing = test_errors)
  }

  raw_error <- unlist(raw_residuals, use.names = FALSE)
  errors <- Reduce("+", err_list) / n_windows

  # Final model on all data (no early stopping here to avoid eval plumbing)
  dall <- xgboost::xgb.DMatrix(data = xmat, label = yvec)
  final_model <- suppressWarnings(
    xgboost::xgb.train(
      params = params,
      data = dall,
      nrounds = as.integer(nrounds),
      verbose = 0
    )
  )

  list(model = final_model, errors = errors, raw_error = raw_error)
}


#' @keywords internal
# -----------------------------
# sequencer (multi-step direct forecasts + residual bootstrap)
# -----------------------------

sequencer <- function(
    seq_len, ts_set, target, deriv, ci = 0.8, min_set = 30, booster = "gbtree",
    max_depth = 4, eta = 1, gamma = 1, min_child_weight = 1, subsample = 0.5, colsample_bytree = 0.7,
    lambda = 1, alpha = 1,
    n_windows = 3, patience = 0.1, nrounds = 100,
    feat_name = "y", dates = NULL, nsim = 1000
) {
  if (!is.data.frame(ts_set)) ts_set <- as.data.frame(ts_set)
  target <- as.numeric(target)

  if (seq_len < 1) stop("sequencer: seq_len must be >= 1")
  if (nrow(ts_set) != length(target)) stop("sequencer: ts_set and target length mismatch.")

  all_positive_check <- all(target >= 0)
  all_negative_check <- all(target <= 0)

  diff_model <- recursive_diff(target, deriv)
  y_diff <- diff_model$vector

  x_aligned <- ts_set
  if (deriv > 0) x_aligned <- x_aligned[(deriv + 1):nrow(x_aligned), , drop = FALSE]
  if (length(y_diff) != nrow(x_aligned)) stop("sequencer: internal alignment mismatch after differencing.")

  # Feasibility for largest horizon
  min_train_for_folds <- 2L * (as.integer(n_windows) + 1L)
  min_train_required  <- max(as.integer(min_set), min_train_for_folds)
  max_seq_feasible <- nrow(x_aligned) - min_train_required
  if (seq_len > max_seq_feasible) stop("sequencer: seq_len too large for available data/windows/min_set.")

  models <- vector("list", seq_len)
  raw_errors <- vector("list", seq_len)
  seq_err_acc <- NULL

  for (h in seq_len(seq_len)) {
    x_train <- x_aligned[seq_len(nrow(x_aligned) - h), , drop = FALSE]
    y_train <- y_diff[(h + 1):length(y_diff)]

    fit <- engine(
      predictors = x_train, target = y_train, booster = booster,
      max_depth = max_depth, eta = eta, gamma = gamma, min_child_weight = min_child_weight,
      subsample = subsample, colsample_bytree = colsample_bytree, lambda = lambda, alpha = alpha,
      n_windows = n_windows, patience = patience, nrounds = nrounds
    )

    models[[h]] <- fit$model
    raw_errors[[h]] <- fit$raw_error
    if (is.null(seq_err_acc)) seq_err_acc <- fit$errors else seq_err_acc <- seq_err_acc + fit$errors
  }
  seq_errors <- seq_err_acc / seq_len

  # --- point forecasts on the last predictor row ---
  x_last <- as.matrix(tail(x_aligned, 1))
  dlast <- xgboost::xgb.DMatrix(data = x_last)

  # IMPORTANT: xgboost::predict is not exported in new API -> use stats::predict
  point_diff <- vapply(
    models,
    function(m) suppressWarnings(as.numeric(stats::predict(m, dlast))),
    numeric(1)
  )

  # --- residual bootstrap simulation in diff-space ---
  sims_diff <- matrix(NA_real_, nrow = nsim, ncol = seq_len)
  for (h in seq_len(seq_len)) {
    re <- raw_errors[[h]]
    if (length(re) < 2 || all(!is.finite(re))) {
      sims_diff[, h] <- point_diff[h]
    } else {
      re <- re[is.finite(re)]
      sims_diff[, h] <- point_diff[h] + sample(re, size = nsim, replace = TRUE)
    }
  }

  sims_level <- .integrate_diffs_matrix(sims_diff, tail_values = diff_model$tail_value)
  if (all_positive_check) sims_level[sims_level < 0] <- 0
  if (all_negative_check) sims_level[sims_level > 0] <- 0

  quants <- sort(unique(c((1 - ci) / 2, 0.25, 0.5, 0.75, ci + (1 - ci) / 2)))
  pred_q <- t(apply(sims_level, 2, function(x) {
    x <- x[is.finite(x)]
    if (!length(x)) return(rep(NA_real_, 6))
    c(min(x), as.numeric(stats::quantile(x, probs = quants, na.rm = TRUE)), max(x))
  }))
  colnames(pred_q) <- c("min", paste0(quants * 100, "%"), "max")

  pred_stats <- t(apply(sims_level, 2, function(x) {
    x <- x[is.finite(x)]
    if (!length(x)) return(c(mean = NA, sd = NA, mode = NA, skewness = NA, kurtosis = NA))
    c(
      mean = mean(x),
      sd = stats::sd(x),
      mode = .mode_density(x),
      skewness = .skewness(x),
      kurtosis = .kurtosis_excess(x)
    )
  }))

  predicted <- as.data.frame(cbind(pred_q, pred_stats))
  predicted <- round(predicted, 6)
  rownames(predicted) <- NULL

  target_level <- invdiff(y_diff, diff_model$head_value)

  if (is.null(dates)) {
    hist_x <- seq_along(target_level)
    for_x <- (length(target_level) + 1):(length(target_level) + seq_len)
  } else {
    hist_x <- tail(dates, length(target_level))
    step <- mean(diff(as.numeric(dates)))
    for_x <- as.Date(tail(dates, 1) + step * seq_len(seq_len))
  }

  lower_col <- paste0(((1 - ci) / 2) * 100, "%")
  upper_col <- paste0((ci + (1 - ci) / 2) * 100, "%")

  plot_obj <- ts_graph_base(
    x_hist = hist_x,
    y_hist = target_level,
    x_forcat = for_x,
    y_forcat = predicted[,"50%"],
    lower = predicted[, lower_col],
    upper = predicted[, upper_col],
    label_x = paste0("Forecast horizon (n=", seq_len, ")"),
    label_y = paste0("Forecast for ", feat_name)
  )

  pred_stats_vec <- c(
    sequential_divergence(t(sims_level)),
    upside_probability(t(sims_level))
  )

  list(models = models, predicted = predicted, seq_errors = seq_errors, pred_stats = pred_stats_vec, plot = plot_obj)
}

#' @keywords internal
# -----------------------------
# hood
# -----------------------------

hood <- function(
    ts_set, seq_len, deriv, norm = TRUE, n_dim,
    ci = 0.8, min_set = 30, booster = "gbtree",
    max_depth = 4, eta = 1, gamma = 1, min_child_weight = 1, subsample = 0.5,
    colsample_bytree = 0.7, lambda = 1, alpha = 1,
    n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL
) {
  if (!is.data.frame(ts_set)) ts_set <- as.data.frame(ts_set)
  orig_set <- ts_set
  feat_names <- colnames(orig_set)
  if (is.null(feat_names)) feat_names <- paste0("V", seq_len(ncol(orig_set)))

  if (length(deriv) == 1) deriv <- rep(as.integer(deriv), ncol(orig_set))
  if (length(deriv) != ncol(orig_set)) stop("hood: `deriv` must have length 1 or ncol(ts_set).")
  deriv <- as.integer(deriv)

  # feasibility guard (largest differencing across targets)
  n_obs <- nrow(orig_set)
  dmax <- max(deriv, na.rm = TRUE)
  n_aligned <- n_obs - dmax

  min_train_for_folds <- 2L * (as.integer(n_windows) + 1L)
  min_train_required  <- max(as.integer(min_set), min_train_for_folds)
  max_seq_feasible <- n_aligned - min_train_required
  min_seq_feasible <- dmax + 1L

  if (!is.finite(max_seq_feasible) || max_seq_feasible < min_seq_feasible) {
    stop("hood: no feasible seq_len for current data/params. Reduce seq_len/min_set/n_windows or differencing.")
  }
  if (seq_len > max_seq_feasible) seq_len <- max_seq_feasible
  if (seq_len < min_seq_feasible) seq_len <- min_seq_feasible

  # predictors pipeline
  X <- orig_set
  if (isTRUE(norm)) {
    X <- as.data.frame(lapply(X, function(x) optimized_yjt(x)$transformed))
  }

  n_feats <- ncol(X)
  if (is.null(n_dim)) n_dim <- n_feats
  n_dim <- max(1L, min(as.integer(n_dim), n_feats))

  if (n_feats > 1 && n_dim < n_feats) {
    s <- base::svd(as.matrix(X))  # <-- explicitly base::svd
    z <- s$u[, seq_len(n_dim), drop = FALSE] %*% diag(s$d[seq_len(n_dim)], n_dim, n_dim)
    X <- as.data.frame(z)
    colnames(X) <- paste0("dim", seq_len(n_dim))
  }

  # forecast each ORIGINAL feature using predictors X
  models <- vector("list", ncol(orig_set))
  for (j in seq_len(ncol(orig_set))) {
    models[[j]] <- sequencer(
      seq_len = seq_len,
      ts_set = X,
      target = orig_set[[j]],       # <-- ORIGINAL target series
      deriv = deriv[j],
      ci = ci,
      min_set = min_set,
      booster = booster,
      max_depth = max_depth,
      eta = eta,
      gamma = gamma,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = colsample_bytree,
      lambda = lambda,
      alpha = alpha,
      n_windows = n_windows,
      patience = patience,
      nrounds = nrounds,
      feat_name = feat_names[j],
      dates = dates
    )
  }

  serie_errors <- lapply(models, function(m) round(m$seq_errors, 6))
  names(serie_errors) <- feat_names

  train_mat <- do.call(rbind, lapply(serie_errors, function(e) e["training", , drop = FALSE]))
  test_mat  <- do.call(rbind, lapply(serie_errors, function(e) e["testing", , drop = FALSE]))
  max_train <- apply(train_mat, 2, max, na.rm = TRUE)
  max_test  <- apply(test_mat, 2, max, na.rm = TRUE)

  joint_error <- rbind(train = max_train, test = max_test)
  colnames(joint_error) <- paste0("max_", colnames(joint_error))

  predictions <- lapply(models, function(m) m$predicted)
  names(predictions) <- feat_names

  if (is.null(dates)) {
    predictions <- lapply(predictions, function(p) { rownames(p) <- paste0("t", seq_len(seq_len)); p })
  } else {
    step <- mean(diff(as.numeric(dates)))
    pred_dates <- as.Date(tail(dates, 1) + step * seq_len(seq_len))
    predictions <- lapply(predictions, function(p) { rownames(p) <- pred_dates; p })
  }

  plots <- lapply(1:length(models), function(m) plot(models[[m]]$plot, y_label = feat_names[m]))
  names(plots) <- feat_names
  pred_stats <- as.data.frame(do.call(rbind, lapply(models, function(m) m$pred_stats)))
  rownames(pred_stats) <- feat_names

  list(
    predictions = predictions,
    joint_error = round(joint_error, 6),
    serie_errors = serie_errors,
    pred_stats = pred_stats,
    plots = plots
  )
}

#' @keywords internal
# -----------------------------
# bayesian_search (new xgboost-API compliant)
# -----------------------------

bayesian_search <- function(
    n_sample = 10, n_search = 5, booster, data, seq_len = NULL, deriv, norm = NULL, n_dim = NULL, ci = 0.8, min_set = 30,
    max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL, lambda = NULL, alpha = NULL,
    n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL, acq = "ucb", kappa = 2.576, eps = 0, kernel = list(type = "exponential", power = 2)
) {
  # ---- Base-R Bayesian-ish optimization (surrogate = lm + UCB over random candidates) ----
  # Keeps signature + output: list(history=..., models=...)
  # Uses ONLY base R inside this function (but still calls your hood(), which uses xgboost etc.)

  if (!is.data.frame(data)) data <- as.data.frame(data)
  n_obs <- nrow(data)
  n_feats <- ncol(data)

  if (length(deriv) == 1) deriv <- rep(as.integer(deriv), n_feats)
  if (length(deriv) != n_feats) stop("bayesian_search: `deriv` must have length 1 or ncol(data).")
  deriv <- as.integer(deriv)

  `%||%` <- function(x, y) if (is.null(x)) y else x

  # --- build tunable bounds (original units) ---
  bounds <- list()

  add_bound <- function(name, rng) {
    rng <- as.numeric(rng)
    if (length(rng) != 2 || any(!is.finite(rng))) return()
    if (rng[1] == rng[2]) return()
    if (rng[2] < rng[1]) rng <- rev(rng)
    bounds[[name]] <<- rng
  }

  # seq_len
  if (is.null(seq_len)) {
    add_bound("seq_len", c(max(deriv) + 1L, max(max(deriv) + 2L, floor(n_obs / 3))))
  } else if (length(seq_len) > 1) {
    add_bound("seq_len", range(seq_len))
  }

  # norm (0/1)
  if (is.null(norm)) add_bound("norm", c(0, 1))

  # n_dim
  if (is.null(n_dim)) {
    add_bound("n_dim", c(1L, n_feats))
  } else if (length(n_dim) > 1) {
    add_bound("n_dim", range(n_dim))
  }

  if (booster == "gbtree") {
    if (is.null(max_depth)) add_bound("max_depth", c(1L, 8L)) else if (length(max_depth) > 1) add_bound("max_depth", range(max_depth))
    if (is.null(eta)) add_bound("eta", c(0.01, 1.0)) else if (length(eta) > 1) add_bound("eta", range(eta))
    if (is.null(gamma)) add_bound("gamma", c(0, 100)) else if (length(gamma) > 1) add_bound("gamma", range(gamma))
    if (is.null(min_child_weight)) add_bound("min_child_weight", c(0, 100)) else if (length(min_child_weight) > 1) add_bound("min_child_weight", range(min_child_weight))
    if (is.null(subsample)) add_bound("subsample", c(0.1, 1.0)) else if (length(subsample) > 1) add_bound("subsample", range(subsample))
    if (is.null(colsample_bytree)) add_bound("colsample_bytree", c(0.1, 1.0)) else if (length(colsample_bytree) > 1) add_bound("colsample_bytree", range(colsample_bytree))
  } else if (booster == "gblinear") {
    if (is.null(eta)) add_bound("eta", c(0.01, 1.0)) else if (length(eta) > 1) add_bound("eta", range(eta))
    if (is.null(lambda)) add_bound("lambda", c(0, 100)) else if (length(lambda) > 1) add_bound("lambda", range(lambda))
    if (is.null(alpha)) add_bound("alpha", c(0, 100)) else if (length(alpha) > 1) add_bound("alpha", range(alpha))
  } else {
    stop("bayesian_search: unknown booster: ", booster)
  }

  if (length(bounds) == 0) stop("bayesian_search: no tunable parameters in bounds (all fixed).")

  # --- utilities: scaling, sampling, rounding/clamping ---
  clamp <- function(x, lo, hi) pmin(hi, pmax(lo, x))

  is_int_param <- function(nm) nm %in% c("seq_len", "n_dim", "max_depth")
  is_bool_param <- function(nm) nm %in% c("norm")

  sample_one <- function() {
    p <- list()
    for (nm in names(bounds)) {
      rng <- bounds[[nm]]
      u <- stats::runif(1)
      v <- rng[1] + u * (rng[2] - rng[1])
      if (is_bool_param(nm)) v <- as.numeric(v) # keep numeric for history build
      if (is_int_param(nm)) v <- round(v)
      p[[nm]] <- v
    }
    p
  }

  normalize_params <- function(p_list) {
    # return numeric named vector in [0,1] for surrogate
    v <- numeric(length(bounds)); names(v) <- names(bounds)
    for (nm in names(bounds)) {
      rng <- bounds[[nm]]
      x <- as.numeric(p_list[[nm]])
      z <- if (rng[2] == rng[1]) 0 else (x - rng[1]) / (rng[2] - rng[1])
      v[nm] <- clamp(z, 0, 1)
    }
    v
  }

  params_to_key <- function(p_list) {
    # stable string key to detect duplicates
    paste(paste(names(p_list), format(p_list, digits = 12, scientific = FALSE), sep = "="), collapse = "|")
  }

  # --- evaluation wrapper (calls hood) ---
  eval_point <- function(p_list) {
    # pull tuned or fixed values
    getp <- function(nm, fixed = NULL) if (!is.null(p_list[[nm]])) p_list[[nm]] else fixed

    seq_len_i <- as.integer(getp("seq_len", fixed = if (!is.null(seq_len) && length(seq_len) == 1) seq_len else max(deriv) + 1L))
    seq_len_i <- max(max(deriv) + 1L, seq_len_i)

    norm_l <- if (!is.null(norm)) as.logical(norm) else as.logical(round(getp("norm", fixed = 0)))
    n_dim_i <- as.integer(getp("n_dim", fixed = if (!is.null(n_dim) && length(n_dim) == 1) n_dim else n_feats))
    n_dim_i <- max(1L, min(n_feats, n_dim_i))

    model <- tryCatch({
      if (booster == "gbtree") {
        hood(
          ts_set = data, seq_len = seq_len_i, deriv = deriv,
          norm = norm_l, n_dim = n_dim_i, ci = ci, min_set = min_set, booster = "gbtree",
          max_depth = as.integer(getp("max_depth", fixed = as.integer(max_depth %||% 4L))),
          eta = as.numeric(getp("eta", fixed = eta %||% 0.3)),
          gamma = as.numeric(getp("gamma", fixed = gamma %||% 0)),
          min_child_weight = as.numeric(getp("min_child_weight", fixed = min_child_weight %||% 1)),
          subsample = as.numeric(getp("subsample", fixed = subsample %||% 0.9)),
          colsample_bytree = as.numeric(getp("colsample_bytree", fixed = colsample_bytree %||% 0.9)),
          lambda = NULL, alpha = NULL,
          n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates
        )
      } else {
        hood(
          ts_set = data, seq_len = seq_len_i, deriv = deriv,
          norm = norm_l, n_dim = n_dim_i, ci = ci, min_set = min_set, booster = "gblinear",
          max_depth = NULL,
          eta = as.numeric(getp("eta", fixed = eta %||% 0.3)),
          gamma = NULL, min_child_weight = NULL, subsample = NULL, colsample_bytree = NULL,
          lambda = as.numeric(getp("lambda", fixed = lambda %||% 1)),
          alpha = as.numeric(getp("alpha", fixed = alpha %||% 0)),
          n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates
        )
      }
    }, error = function(e) NULL)

    if (is.null(model) || is.null(model$joint_error) || !is.matrix(model$joint_error) || !"test" %in% rownames(model$joint_error)) {
      return(list(ok = FALSE, score = -1e6, model = NULL, test_err = NULL))
    }

    cols <- intersect(colnames(model$joint_error), c("max_rmse", "max_mae", "max_mdae"))
    if (length(cols) < 1) cols <- colnames(model$joint_error)[seq_len(min(3L, ncol(model$joint_error)))]

    err <- mean(as.numeric(model$joint_error["test", cols]), na.rm = TRUE)
    if (!is.finite(err)) return(list(ok = FALSE, score = -1e6, model = NULL, test_err = NULL))

    test_err <- as.numeric(model$joint_error["test", , drop = TRUE])
    names(test_err) <- colnames(model$joint_error)

    list(ok = TRUE, score = -err, model = model, test_err = test_err)
  }

  # --- weighted rank helper (base R) ---
  if (!exists(".wgt_rank", mode = "function")) {
    .wgt_rank <- function(err_mat) {
      # err_mat: matrix, rows=models, cols=error metrics (lower is better)
      err_mat <- as.matrix(err_mat)
      if (nrow(err_mat) <= 1) return(rep(1, nrow(err_mat)))
      w <- apply(err_mat, 2, function(x) {
        x <- x[is.finite(x)]
        if (length(x) < 2) return(NA_real_)
        mu <- mean(x); s <- stats::sd(x)
        if (!is.finite(mu) || !is.finite(s) || mu == 0) return(NA_real_)
        abs(s / mu)
      })
      finite_w <- is.finite(w) & w > 0
      if (!any(finite_w)) return(rowMeans(apply(abs(err_mat), 2, rank, ties.method = "average"), na.rm = TRUE))
      r <- apply(abs(err_mat[, finite_w, drop = FALSE]), 2, rank, ties.method = "average")
      as.numeric(apply(r, 1, weighted.mean, w = w[finite_w], na.rm = TRUE))
    }
  }

  # ---- optimization loop ----
  total_evals <- max(1L, as.integer(n_sample + n_search))

  params_list <- vector("list", total_evals)
  scores <- rep(NA_real_, total_evals)
  test_errs <- vector("list", total_evals)
  models <- vector("list", total_evals)

  seen <- new.env(parent = emptyenv())

  # initial random points
  i <- 1L
  while (i <= min(n_sample, total_evals)) {
    p <- sample_one()
    key <- params_to_key(p)
    if (exists(key, envir = seen, inherits = FALSE)) next
    assign(key, TRUE, envir = seen)
    res <- eval_point(p)

    params_list[[i]] <- p
    scores[i] <- res$score
    test_errs[[i]] <- res$test_err
    models[[i]] <- res$model
    i <- i + 1L
  }

  # sequential BO-ish steps: fit lm surrogate on successful points, pick next by UCB over random candidates
  for (t in (i):total_evals) {
    ok_idx <- which(is.finite(scores) & scores > -1e5 & !vapply(models, is.null, logical(1)))
    if (length(ok_idx) < 3L) {
      # not enough to fit surrogate: keep random
      p <- sample_one()
    } else {
      X <- do.call(rbind, lapply(params_list[ok_idx], normalize_params))
      y <- scores[ok_idx]

      # simple quadratic features (still base R)
      Xdf <- as.data.frame(X)
      for (nm in names(Xdf)) Xdf[[paste0(nm, "_2")]] <- Xdf[[nm]]^2

      fit <- tryCatch(stats::lm(y ~ ., data = Xdf), error = function(e) NULL)

      # propose among candidates
      n_cand <- 250L
      cand <- vector("list", n_cand)
      candX <- matrix(NA_real_, nrow = n_cand, ncol = length(bounds))
      colnames(candX) <- names(bounds)

      c_ok <- 0L
      tries <- 0L
      while (c_ok < n_cand && tries < n_cand * 20L) {
        tries <- tries + 1L
        pp <- sample_one()
        key <- params_to_key(pp)
        if (exists(key, envir = seen, inherits = FALSE)) next
        c_ok <- c_ok + 1L
        cand[[c_ok]] <- pp
        candX[c_ok, ] <- normalize_params(pp)
      }
      if (c_ok < 1L) {
        p <- sample_one()
      } else {
        candX <- candX[seq_len(c_ok), , drop = FALSE]
        canddf <- as.data.frame(candX)
        for (nm in names(canddf)) canddf[[paste0(nm, "_2")]] <- canddf[[nm]]^2

        if (is.null(fit)) {
          p <- cand[[1L]]
        } else {
          mu <- as.numeric(suppressWarnings(stats::predict(fit, newdata = canddf)))
          # crude uncertainty: use residual sd (constant) -> still useful for exploration
          s <- stats::sigma(fit)
          if (!is.finite(s) || s <= 0) s <- 0
          ucb <- mu + as.numeric(kappa) * s
          best <- which.max(ucb)
          p <- cand[[best]]
        }
      }
    }

    key <- params_to_key(p)
    if (!exists(key, envir = seen, inherits = FALSE)) assign(key, TRUE, envir = seen)
    res <- eval_point(p)

    params_list[[t]] <- p
    scores[t] <- res$score
    test_errs[[t]] <- res$test_err
    models[[t]] <- res$model
  }

  # ---- post-process: keep successful only (like your robust GP-filtering) ----
  ok <- vapply(models, function(m) {
    is.list(m) && !is.null(m$joint_error) && is.matrix(m$joint_error) && "test" %in% rownames(m$joint_error)
  }, logical(1))

  if (!any(ok)) stop("bayesian_search: all trials failed (all Pred are NULL/invalid).")

  params_ok <- params_list[ok]
  models_ok <- models[ok]
  test_ok <- test_errs[ok]

  # Build param history table
  par_names <- unique(c(
    if (booster == "gbtree") c("seq_len","norm","n_dim","max_depth","eta","gamma","min_child_weight","subsample","colsample_bytree")
    else c("seq_len","norm","n_dim","eta","lambda","alpha")
  ))

  par_mat <- matrix(NA_real_, nrow = length(params_ok), ncol = length(par_names))
  colnames(par_mat) <- par_names
  for (r in seq_along(params_ok)) {
    p <- params_ok[[r]]
    for (nm in names(p)) if (nm %in% par_names) par_mat[r, nm] <- as.numeric(p[[nm]])
  }

  # Fill fixed values that were not tuned (so history is complete)
  fill_fixed <- function(nm, val) {
    if (!nm %in% colnames(par_mat)) return()
    na_idx <- which(!is.finite(par_mat[, nm]))
    if (length(na_idx)) par_mat[na_idx, nm] <<- as.numeric(val)
  }

  if (!is.null(seq_len) && length(seq_len) == 1) fill_fixed("seq_len", seq_len)
  if (!is.null(norm)) fill_fixed("norm", as.numeric(norm))
  if (!is.null(n_dim) && length(n_dim) == 1) fill_fixed("n_dim", n_dim)

  if (booster == "gbtree") {
    if (!is.null(max_depth) && length(max_depth) == 1) fill_fixed("max_depth", max_depth)
    if (!is.null(eta) && length(eta) == 1) fill_fixed("eta", eta)
    if (!is.null(gamma) && length(gamma) == 1) fill_fixed("gamma", gamma)
    if (!is.null(min_child_weight) && length(min_child_weight) == 1) fill_fixed("min_child_weight", min_child_weight)
    if (!is.null(subsample) && length(subsample) == 1) fill_fixed("subsample", subsample)
    if (!is.null(colsample_bytree) && length(colsample_bytree) == 1) fill_fixed("colsample_bytree", colsample_bytree)
  } else {
    if (!is.null(eta) && length(eta) == 1) fill_fixed("eta", eta)
    if (!is.null(lambda) && length(lambda) == 1) fill_fixed("lambda", lambda)
    if (!is.null(alpha) && length(alpha) == 1) fill_fixed("alpha", alpha)
  }

  # error matrix from joint_error test rows
  err_mat <- do.call(rbind, lapply(models_ok, function(m) as.numeric(m$joint_error["test", , drop = TRUE])))
  colnames(err_mat) <- colnames(models_ok[[1]]$joint_error)

  wgt_avg_rank <- round(.wgt_rank(err_mat), 4)

  history <- data.frame(par_mat, err_mat, wgt_avg_rank = wgt_avg_rank, check.names = FALSE)
  if ("norm" %in% names(history)) history$norm <- as.logical(round(history$norm))
  rownames(history) <- NULL

  # ensure models are named exactly like expected downstream
  models_ok <- lapply(models_ok, function(m) {
    if (is.null(names(m))) {
      names(m) <- c("predictions","joint_error","serie_errors","pred_stats","plots")
    } else {
      # keep as-is (your hood already names these)
    }
    m
  })

  list(history = history, models = models_ok)
}


#' @keywords internal
# -----------------------------
# random_search
# -----------------------------

random_search <- function(
    n_sample, data, booster = "gbtree", seq_len = NULL, deriv,
    norm = NULL, n_dim = NULL, ci = 0.8, min_set = 30,
    max_depth = NULL, eta = NULL, gamma = NULL, min_child_weight = NULL,
    subsample = NULL, colsample_bytree = NULL, lambda = NULL, alpha = NULL,
    n_windows = 3, patience = 0.1, nrounds = 100, dates = NULL, seed = 42
) {
  set.seed(seed)
  n_obs <- nrow(data)
  n_feats <- ncol(data)

  # seq_len candidates
  if (is.null(seq_len)) {
    sl_set <- sample.int(as.integer(floor(n_obs / 3)), n_sample, replace = TRUE)
  } else if (length(seq_len) > 1) {
    sl_set <- sample(seq_len, n_sample, replace = TRUE)
  } else {
    sl_set <- rep(as.integer(seq_len), n_sample)
  }

  # clamp seq_len to feasible range (based on max differencing across targets)
  dmax <- if (length(deriv) > 0) max(as.integer(deriv), na.rm = TRUE) else 0L
  n_aligned <- n_obs - dmax
  min_train_for_folds <- 2L * (as.integer(n_windows) + 1L)
  min_train_required  <- max(as.integer(min_set), min_train_for_folds)
  max_seq_feasible <- n_aligned - min_train_required
  min_seq_feasible <- dmax + 1L

  if (!is.finite(max_seq_feasible) || max_seq_feasible < min_seq_feasible) {
    stop("random_search: no feasible seq_len given current data/params. ",
         "Need n_obs - max(deriv) > min_set + 2*(n_windows+1).")
  }

  sl_set <- pmax(sl_set, min_seq_feasible)
  sl_set <- pmin(sl_set, max_seq_feasible)

  # norm candidates
  if (is.null(norm)) norm_set <- sample(c(TRUE, FALSE), n_sample, replace = TRUE) else norm_set <- rep(as.logical(norm), n_sample)

  # n_dim candidates
  if (is.null(n_dim)) {
    dim_set <- sample.int(n_feats, n_sample, replace = TRUE)
  } else if (length(n_dim) > 1) {
    dim_set <- sample(n_dim, n_sample, replace = TRUE)
  } else {
    dim_set <- rep(as.integer(n_dim), n_sample)
  }
  dim_set[dim_set > n_feats] <- n_feats

  exploration <- vector("list", n_sample)
  err_msg <- character(n_sample)

  if (booster == "gbtree") {
    depth_set <- if (is.null(max_depth)) sample.int(8L, n_sample, replace = TRUE) else if (length(max_depth) > 1) sample(max_depth, n_sample, replace = TRUE) else rep(as.integer(max_depth), n_sample)
    eta_set   <- if (is.null(eta)) runif(n_sample, 0.01, 1.0) else if (length(eta) > 1) sample(eta, n_sample, replace = TRUE) else rep(eta, n_sample)
    gamma_set <- if (is.null(gamma)) runif(n_sample, 0, 100) else if (length(gamma) > 1) sample(gamma, n_sample, replace = TRUE) else rep(gamma, n_sample)
    mcw_set   <- if (is.null(min_child_weight)) runif(n_sample, 0, 100) else if (length(min_child_weight) > 1) sample(min_child_weight, n_sample, replace = TRUE) else rep(min_child_weight, n_sample)
    ss_set    <- if (is.null(subsample)) runif(n_sample, 0.1, 1.0) else if (length(subsample) > 1) sample(subsample, n_sample, replace = TRUE) else rep(subsample, n_sample)
    csbt_set  <- if (is.null(colsample_bytree)) runif(n_sample, 0.1, 1.0) else if (length(colsample_bytree) > 1) sample(colsample_bytree, n_sample, replace = TRUE) else rep(colsample_bytree, n_sample)

    for (i in seq_len(n_sample)) {
      exploration[[i]] <- tryCatch(
        hood(
          ts_set = data, seq_len = sl_set[i], deriv = deriv,
          norm = norm_set[i], n_dim = dim_set[i], ci = ci, min_set = min_set, booster = "gbtree",
          max_depth = as.integer(depth_set[i]), eta = eta_set[i], gamma = gamma_set[i], min_child_weight = mcw_set[i],
          subsample = ss_set[i], colsample_bytree = csbt_set[i],
          lambda = NULL, alpha = NULL, n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates
        ),
        error = function(e) { err_msg[i] <<- conditionMessage(e); NULL }
      )
    }

    keep <- vapply(exploration, function(x) !is.null(x), logical(1))
    if (!any(keep)) {
      msg <- err_msg[nzchar(err_msg)]
      stop("random_search: all candidate models failed. Examples:\n- ",
           paste(utils::head(msg, 8), collapse = "\n- "))
    }

    exploration <- exploration[keep]
    sl_set <- sl_set[keep]; norm_set <- norm_set[keep]; dim_set <- dim_set[keep]
    depth_set <- depth_set[keep]; eta_set <- eta_set[keep]; gamma_set <- gamma_set[keep]
    mcw_set <- mcw_set[keep]; ss_set <- ss_set[keep]; csbt_set <- csbt_set[keep]

    test_errs <- do.call(rbind, lapply(exploration, function(m) m$joint_error["test", , drop = TRUE]))
    test_errs <- as.matrix(test_errs)
    wgt_avg_rank <- round(.wgt_rank(test_errs), 4)

    history <- cbind(
      data.frame(
        seq_len = sl_set,
        norm = norm_set,
        n_dim = dim_set,
        max_depth = depth_set,
        eta = round(eta_set, 6),
        gamma = round(gamma_set, 6),
        min_child_weight = round(mcw_set, 6),
        subsample = round(ss_set, 6),
        colsample_bytree = round(csbt_set, 6),
        stringsAsFactors = FALSE
      ),
      round(test_errs, 6),
      wgt_avg_rank = wgt_avg_rank
    )

  } else if (booster == "gblinear") {
    eta_set    <- if (is.null(eta)) runif(n_sample, 0.01, 1.0) else if (length(eta) > 1) sample(eta, n_sample, replace = TRUE) else rep(eta, n_sample)
    lambda_set <- if (is.null(lambda)) runif(n_sample, 0, 100) else if (length(lambda) > 1) sample(lambda, n_sample, replace = TRUE) else rep(lambda, n_sample)
    alpha_set  <- if (is.null(alpha)) runif(n_sample, 0, 100) else if (length(alpha) > 1) sample(alpha, n_sample, replace = TRUE) else rep(alpha, n_sample)

    for (i in seq_len(n_sample)) {
      exploration[[i]] <- tryCatch(
        hood(
          ts_set = data, seq_len = sl_set[i], deriv = deriv,
          norm = norm_set[i], n_dim = dim_set[i], ci = ci, min_set = min_set, booster = "gblinear",
          max_depth = NULL, eta = eta_set[i], gamma = NULL, min_child_weight = NULL,
          subsample = NULL, colsample_bytree = NULL, lambda = lambda_set[i], alpha = alpha_set[i],
          n_windows = n_windows, patience = patience, nrounds = nrounds, dates = dates
        ),
        error = function(e) { err_msg[i] <<- conditionMessage(e); NULL }
      )
    }

    keep <- vapply(exploration, function(x) !is.null(x), logical(1))
    if (!any(keep)) {
      msg <- err_msg[nzchar(err_msg)]
      stop("random_search: all candidate models failed. Examples:\n- ",
           paste(utils::head(msg, 8), collapse = "\n- "))
    }

    exploration <- exploration[keep]
    sl_set <- sl_set[keep]; norm_set <- norm_set[keep]; dim_set <- dim_set[keep]
    eta_set <- eta_set[keep]; lambda_set <- lambda_set[keep]; alpha_set <- alpha_set[keep]

    test_errs <- do.call(rbind, lapply(exploration, function(m) m$joint_error["test", , drop = TRUE]))
    test_errs <- as.matrix(test_errs)
    wgt_avg_rank <- round(.wgt_rank(test_errs), 4)

    history <- cbind(
      data.frame(
        seq_len = sl_set,
        norm = norm_set,
        n_dim = dim_set,
        eta = round(eta_set, 6),
        lambda = round(lambda_set, 6),
        alpha = round(alpha_set, 6),
        stringsAsFactors = FALSE
      ),
      round(test_errs, 6),
      wgt_avg_rank = wgt_avg_rank
    )

  } else stop("Unknown booster: ", booster)

  rownames(history) <- NULL
  list(history = history, models = exploration)
}

#' @keywords internal
# -----------------------------
# Weighted rank
# -----------------------------

.wgt_rank <- function(test_errs) {
  test_errs <- as.matrix(test_errs)
  if (is.null(test_errs) || length(test_errs) == 0 || nrow(test_errs) == 0 || ncol(test_errs) == 0) return(numeric(0))
  if (nrow(test_errs) == 1) return(1)

  weights <- apply(test_errs, 2, function(x) {
    x <- x[is.finite(x)]
    if (length(x) < 2) return(NA_real_)
    mu <- mean(x)
    if (!is.finite(mu) || mu == 0) return(NA_real_)
    abs(stats::sd(x) / mu)
  })

  finite_w <- is.finite(weights) & weights > 0
  if (!any(finite_w)) return(rep(1, nrow(test_errs)))

  ranks <- apply(abs(test_errs[, finite_w, drop = FALSE]), 2, rank, ties.method = "average")
  as.numeric(apply(ranks, 1, weighted.mean, w = weights[finite_w]))
}

#' @keywords internal
# -----------------------------
# Base plot object
# -----------------------------
#' @keywords internal
ts_graph_base <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL,
                          label_x = "Horizon", label_y = "Forecast") {
  structure(
    list(x_hist = x_hist, y_hist = y_hist,
         x_forcat = x_forcat, y_forcat = y_forcat,
         lower = lower, upper = upper,
         label_x = label_x, label_y = label_y),
    class = "audrex_plot"
  )
}

#' @keywords internal
#' Plot method for audrex_plot that SAVES to PNG/BMP (base graphics)
#'
#' This version actually DRAWS something even if your audrex_plot object is a plain list.
#' It supports:
#'  - a single audrex_plot object
#'  - a list of audrex_plot objects (e.g., best_model$plots)
#'
#' Expected fields inside an audrex_plot object (best-effort; missing ones are handled):
#'   x_hist, y_hist, x_forcat, y_forcat, lower, upper,
#'   label_x, label_y, main
#'
plot.audrex_plot <- function(
    x,
    file = NULL,
    device = c("png", "bmp"),
    width = 1600,
    height = 900,
    res = 150,
    units = "px",
    one_file_per_plot = FALSE,
    y_label = NULL,   # <-- NEW: override y-axis label
    ...
) {
  device <- match.arg(device)
  `%||%` <- function(a, b) if (is.null(a)) b else a

  .is_date <- function(v) inherits(v, "Date") || inherits(v, "POSIXt")

  .as_num_x <- function(v) {
    if (is.null(v)) return(NULL)
    if (.is_date(v)) return(as.numeric(v))
    suppressWarnings(as.numeric(v))
  }

  .open_dev <- function(path) {
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    if (device == "png") {
      grDevices::png(filename = path, width = width, height = height, res = res, units = units)
    } else {
      grDevices::bmp(filename = path, width = width, height = height, units = units)
    }
  }

  .close_dev <- function() {
    if (grDevices::dev.cur() != 1L) grDevices::dev.off()
  }

  .draw_one <- function(p) {
    if (is.list(p) && is.function(p$draw)) {
      p$draw(...)
      return(invisible(NULL))
    }

    x_hist   <- p$x_hist %||% p$xh %||% p$hist_x %||% NULL
    y_hist   <- p$y_hist %||% p$yh %||% p$hist_y %||% NULL
    x_forcat <- p$x_forcat %||% p$xf %||% p$forcat_x %||% p$x_forecast %||% NULL
    y_forcat <- p$y_forcat %||% p$yf %||% p$forcat_y %||% p$y_forecast %||% NULL
    lower    <- p$lower %||% p$lo %||% p$y_lower %||% NULL
    upper    <- p$upper %||% p$hi %||% p$y_upper %||% NULL

    label_x <- p$label_x %||% "Horizon"
    label_y <- if (!is.null(y_label) && is.character(y_label) && nzchar(y_label)) y_label else (p$label_y %||% "Value")
    main    <- p$main %||% p$title %||% ""

    if (is.null(x_hist) || is.null(y_hist) || is.null(x_forcat) || is.null(y_forcat)) {
      plot.new()
      text(0.5, 0.5, "audrex_plot: missing fields\n(x_hist/y_hist/x_forcat/y_forcat)", cex = 0.9)
      return(invisible(NULL))
    }

    xh_num <- .as_num_x(x_hist)
    xf_num <- .as_num_x(x_forcat)
    yh <- as.numeric(y_hist)
    yf <- as.numeric(y_forcat)

    yy <- c(yh, yf)
    if (!is.null(lower)) yy <- c(yy, as.numeric(lower))
    if (!is.null(upper)) yy <- c(yy, as.numeric(upper))
    yy <- yy[is.finite(yy)]

    xx <- c(xh_num, xf_num)
    xx <- xx[is.finite(xx)]

    if (length(xx) < 2 || length(yy) < 2) {
      plot.new()
      text(0.5, 0.5, "audrex_plot: insufficient finite data to plot", cex = 0.9)
      return(invisible(NULL))
    }

    plot(
      xh_num, yh,
      type = "l",
      xlim = range(xx, na.rm = TRUE),
      ylim = range(yy, na.rm = TRUE),
      xlab = label_x,
      ylab = label_y,
      main = main,
      ...
    )

    if (!is.null(lower) && !is.null(upper)) {
      lo <- as.numeric(lower)
      hi <- as.numeric(upper)
      ok <- is.finite(xf_num) & is.finite(lo) & is.finite(hi)
      if (any(ok)) {
        xf_ok <- xf_num[ok]
        polygon(
          x = c(xf_ok, rev(xf_ok)),
          y = c(lo[ok], rev(hi[ok])),
          border = NA,
          col = grDevices::adjustcolor("gray70", alpha.f = 0.5)
        )
      }
    }

    lines(xf_num, yf, lwd = 2)
    abline(v = min(xf_num, na.rm = TRUE), lty = 3)

    if (.is_date(x_hist) || .is_date(x_forcat)) {
      all_x <- c(x_hist, x_forcat)
      at_num <- .as_num_x(all_x)
      ok <- is.finite(at_num)
      if (any(ok)) {
        idx <- unique(round(seq(1, sum(ok), length.out = min(6, sum(ok)))))
        at <- at_num[ok][idx]
        labs <- format(all_x[ok][idx])
        axis(1, at = at, labels = labs)
      }
    }

    invisible(NULL)
  }

  is_plot_list <- is.list(x) && !inherits(x, "audrex_plot")

  # ---- SAVE MODE ----
  if (!is.null(file)) {
    if (!is.character(file) || length(file) != 1L || !nzchar(file)) {
      stop("plot.audrex_plot: `file` must be a non-empty file path.")
    }

    if (is_plot_list) {
      plots <- x
      n <- length(plots)
      if (n == 0L) stop("plot.audrex_plot: empty plot list.")

      if (one_file_per_plot) {
        base <- sub("\\.(png|bmp)$", "", file, ignore.case = TRUE)
        ext <- if (grepl("\\.(png|bmp)$", file, ignore.case = TRUE)) sub("^.*\\.(png|bmp)$", "\\1", file, ignore.case = TRUE) else device

        for (i in seq_len(n)) {
          path_i <- sprintf("%s_%03d.%s", base, i, ext)
          .open_dev(path_i)
          on.exit(.close_dev(), add = TRUE)
          .draw_one(plots[[i]])
          .close_dev()
        }
        return(invisible(file))
      }

      .open_dev(file)
      on.exit(.close_dev(), add = TRUE)

      old_par <- graphics::par(no.readonly = TRUE)
      on.exit(graphics::par(old_par), add = TRUE)

      ncol <- ceiling(sqrt(n))
      nrow <- ceiling(n / ncol)
      graphics::par(mfrow = c(nrow, ncol), mar = c(4, 4, 3, 1) + 0.1)

      for (i in seq_len(n)) .draw_one(plots[[i]])

      .close_dev()
      return(invisible(file))
    }

    .open_dev(file)
    on.exit(.close_dev(), add = TRUE)
    .draw_one(x)
    .close_dev()
    return(invisible(file))
  }

  # ---- LIVE MODE ----
  if (is_plot_list) {
    plots <- x
    n <- length(plots)
    if (n == 0L) stop("plot.audrex_plot: empty plot list.")

    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par), add = TRUE)

    ncol <- ceiling(sqrt(n))
    nrow <- ceiling(n / ncol)
    graphics::par(mfrow = c(nrow, ncol), mar = c(4, 4, 3, 1) + 0.1)

    for (i in seq_len(n)) .draw_one(plots[[i]])
    return(invisible(NULL))
  }

  .draw_one(x)

  p <- recordPlot()
  return(p)
}


# -----------------------------
# Helpers
# -----------------------------

#' @keywords internal
recursive_diff <- function(vector, deriv) {
  vector <- as.numeric(vector)
  head_value <- numeric(0)
  tail_value <- numeric(0)
  if (deriv <= 0) return(list(vector = vector, head_value = NULL, tail_value = NULL))
  for (i in seq_len(deriv)) {
    head_value[i] <- vector[1]
    tail_value[i] <- vector[length(vector)]
    vector <- diff(vector)
  }
  list(vector = vector, head_value = head_value, tail_value = tail_value)
}

#' @keywords internal
invdiff <- function(vector, heads, add = FALSE) {
  vector <- as.numeric(vector)
  if (is.null(heads) || !length(heads)) return(vector)
  for (d in length(heads):1) vector <- cumsum(c(heads[d], vector))
  if (!add) return(vector[-seq_len(length(heads))])
  vector
}

#' @keywords internal
.integrate_diffs_matrix <- function(diffs, tail_values) {
  if (is.null(tail_values)) return(diffs)
  out <- diffs
  for (d in length(tail_values):1) {
    base <- tail_values[d]
    out <- t(apply(out, 1, function(v) {
      vv <- cumsum(c(base, v))
      vv[-1]
    }))
  }
  out
}

#' @keywords internal
best_deriv <- function(ts, max_diff = 3, thresh = 0.001) {
  ts <- as.numeric(ts)
  pvalues <- rep(NA_real_, max_diff + 1L)
  cur <- ts
  for (d in seq_len(max_diff + 1L)) {
    t <- seq_along(cur)
    model <- stats::lm(cur ~ t)
    fs <- summary(model)$fstatistic
    pvalues[d] <- stats::pf(fs[1], fs[2], fs[3], lower.tail = FALSE)
    if (length(cur) < 3) break
    cur <- diff(cur)
  }
  sum(pvalues < thresh, na.rm = TRUE)
}

#' @keywords internal
optimized_yjt <- function(vector, precision = 100) {
  x <- as.numeric(vector)

  yjt_fun <- function(x, lambda) {
    out <- numeric(length(x))
    pos <- x >= 0
    neg <- !pos
    if (lambda != 0) out[pos] <- ((x[pos] + 1)^lambda - 1) / lambda else out[pos] <- log(x[pos] + 1)
    if (lambda != 2) out[neg] <- -((( -x[neg] + 1)^(2 - lambda) - 1) / (2 - lambda)) else out[neg] <- -log(-x[neg] + 1)
    out
  }

  lambda_seq <- seq(0, 2, length.out = precision)
  cor_seq <- vapply(lambda_seq, function(lam) stats::cor(yjt_fun(x, lam), x), numeric(1))
  best_lambda <- lambda_seq[which.max(cor_seq)]

  list(
    transformed = yjt_fun(x, best_lambda),
    best_lambda = best_lambda,
    direct_fun  = function(z) yjt_fun(z, best_lambda)
  )
}

#' @keywords internal
.reg_errors <- function(y, yhat) {
  y <- as.numeric(y); yhat <- as.numeric(yhat)
  e <- y - yhat
  rmse <- sqrt(mean(e^2, na.rm = TRUE))
  mae  <- mean(abs(e), na.rm = TRUE)
  mdae <- stats::median(abs(e), na.rm = TRUE)

  denom <- abs(y)
  ok <- is.finite(denom) & denom > 0
  mape <- if (any(ok)) mean(abs(e[ok]) / denom[ok], na.rm = TRUE) else NA_real_

  d <- diff(y)
  mase_den <- mean(abs(d), na.rm = TRUE)
  mase <- if (is.finite(mase_den) && mase_den > 0) mae / mase_den else NA_real_

  ybar <- mean(y, na.rm = TRUE)
  denom_ae <- mean(abs(y - ybar), na.rm = TRUE)
  rae <- if (is.finite(denom_ae) && denom_ae > 0) mae / denom_ae else NA_real_

  denom_se <- mean((y - ybar)^2, na.rm = TRUE)
  rse <- if (is.finite(denom_se) && denom_se > 0) mean(e^2, na.rm = TRUE) / denom_se else NA_real_
  rrse <- if (is.finite(rse) && rse >= 0) sqrt(rse) else NA_real_

  c(rmse = rmse, mae = mae, mdae = mdae, mape = mape, mase = mase, rae = rae, rse = rse, rrse = rrse)
}

#' @keywords internal
upside_probability <- function(m) {
  mat <- as.matrix(m)
  if (nrow(mat) < 2 && ncol(mat) >= 2) mat <- t(mat)
  n <- nrow(mat)
  if (n < 2) return(c(avg_upside_prob = NA_real_, terminal_upside_prob = NA_real_))

  growths <- mat[-1, , drop = FALSE] / mat[-n, , drop = FALSE] - 1
  avg_upp <- mean(growths > 0, na.rm = TRUE)

  terminal_growth <- mat[n, ] / mat[1, ] - 1
  terminal_upp <- mean(terminal_growth > 0, na.rm = TRUE)

  c(avg_upside_prob = round(avg_upp, 6), terminal_upside_prob = round(terminal_upp, 6))
}

#' @keywords internal
sequential_divergence <- function(m) {
  mat <- as.matrix(m)
  if (nrow(mat) < 2 && ncol(mat) >= 2) mat <- t(mat)
  n <- nrow(mat)
  if (n < 2) return(c(avg_divergence = NA_real_, terminal_divergence = NA_real_))

  rng <- range(mat, finite = TRUE)
  if (!all(is.finite(rng)) || diff(rng) == 0) return(c(avg_divergence = 0, terminal_divergence = 0))
  s <- seq(rng[1], rng[2], length.out = 200)

  ecdfs <- lapply(seq_len(n), function(i) stats::ecdf(mat[i, ]))
  seq_div <- numeric(n - 1)
  for (i in 2:n) seq_div[i - 1] <- max(abs(ecdfs[[i]](s) - ecdfs[[i - 1]](s)))

  c(avg_divergence = round(mean(seq_div, na.rm = TRUE), 6),
    terminal_divergence = round(max(abs(ecdfs[[n]](s) - ecdfs[[1]](s))), 6))
}

#' @keywords internal
.safe_nthread <- function() {
  n <- 1L
  if (requireNamespace("parallel", quietly = TRUE)) {
    n <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) 1L)
    n <- max(1L, as.integer(n) - 1L)
  }
  n
}

#' @keywords internal
.as_early_stopping_rounds <- function(patience, nrounds) {
  if (length(patience) != 1) return(10L)
  if (is.numeric(patience) && patience > 0 && patience < 1) return(max(1L, as.integer(floor(patience * nrounds))))
  if (is.numeric(patience) && patience >= 1) return(as.integer(round(patience)))
  10L
}

#' @keywords internal
.make_time_folds <- function(n, n_windows) {
  if (n_windows < 1) stop("n_windows must be >= 1")
  block <- floor(n / (n_windows + 1))
  if (block < 2) stop("Too few observations for the requested n_windows.")
  folds <- vector("list", n_windows)
  for (w in seq_len(n_windows)) {
    train_end <- w * block
    test_start <- train_end + 1
    test_end <- min((w + 1) * block, n)
    if (test_start > n) test_start <- n
    if (test_end < test_start) test_end <- test_start
    folds[[w]] <- list(train = seq_len(train_end), test = test_start:test_end)
  }
  folds
}

#' @keywords internal
.format_seconds <- function(sec) {
  sec <- as.numeric(sec)
  if (!is.finite(sec)) return(NA_character_)
  h <- floor(sec / 3600); sec <- sec - 3600 * h
  m <- floor(sec / 60);   s <- round(sec - 60 * m)
  sprintf("%02dh:%02dm:%02ds", h, m, s)
}

#' @keywords internal
.safe_mean <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  mean(x)
}

#' @keywords internal
.mode_density <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  d <- stats::density(x)
  d$x[which.max(d$y)]
}

#' @keywords internal
.skewness <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) return(NA_real_)
  m <- mean(x); s <- stats::sd(x)
  if (!is.finite(s) || s == 0) return(0)
  mean(((x - m) / s)^3)
}

#' @keywords internal
.kurtosis_excess <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 4) return(NA_real_)
  m <- mean(x); s <- stats::sd(x)
  if (!is.finite(s) || s == 0) return(0)
  mean(((x - m) / s)^4) - 3
}

#' @keywords internal
.optimal_loess_smooth <- function(y, spans = c(0.15, 0.25, 0.35, 0.5, 0.7)) {
  y <- as.numeric(y)
  n <- length(y)
  x <- seq_len(n)
  best <- y
  best_sse <- Inf
  for (sp in spans) {
    fit <- tryCatch(stats::loess(y ~ x, span = sp, degree = 2, family = "gaussian"), error = function(e) NULL)
    if (is.null(fit)) next
    yhat <- tryCatch(stats::predict(fit, x), error = function(e) rep(NA_real_, n))
    if (all(!is.finite(yhat))) next
    sse <- sum((y - yhat)^2, na.rm = TRUE)
    if (is.finite(sse) && sse < best_sse) { best_sse <- sse; best <- yhat }
  }
  best
}

