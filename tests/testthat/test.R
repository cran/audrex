
# -------------------------------
# tests/testthat/helper-data.R
# -------------------------------
.make_ts_df <- function(n = 140, p = 4, seed = 1) {
  set.seed(seed)
  base <- cumsum(rnorm(n))
  df <- data.frame(
    A = base + rnorm(n, sd = 0.2),
    B = 0.7 * base + cumsum(rnorm(n, sd = 0.8)),
    C = cumsum(rnorm(n, sd = 1.1)),
    D = cumsum(rnorm(n, sd = 0.9))
  )
  df[, seq_len(p), drop = FALSE]
}

.make_df_with_na <- function(n = 120, seed = 1) {
  df <- .make_ts_df(n = n, p = 3, seed = seed)
  set.seed(seed + 10)
  df[sample.int(n, 6), 1] <- NA
  df[sample.int(n, 4), 2] <- NA
  df
}

.expect_model_bundle <- function(m) {
  testthat::expect_true(is.list(m))
  testthat::expect_true(all(c("predictions", "joint_error", "serie_errors", "pred_stats", "plots") %in% names(m)))
  testthat::expect_true(is.list(m$predictions))
  testthat::expect_true(is.matrix(m$joint_error))
}

# -------------------------------
# tests/testthat/test-engine.R
# -------------------------------
testthat::test_that("engine works for gbtree and returns proper structure", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 120, p = 4, seed = 11)
  y <- df[[1]]
  X <- df[, -1, drop = FALSE]

  out <- engine(
    predictors = X, target = y, booster = "gbtree",
    max_depth = 3, eta = 0.2, gamma = 0, min_child_weight = 1,
    subsample = 0.9, colsample_bytree = 0.9,
    lambda = 1, alpha = 0,
    n_windows = 2, patience = 0.2, nrounds = 20
  )

  testthat::expect_true(is.list(out))
  testthat::expect_true(all(c("model", "errors", "raw_error") %in% names(out)))
  testthat::expect_true(inherits(out$model, "xgb.Booster"))
  testthat::expect_true(is.matrix(out$errors))
  testthat::expect_equal(dim(out$errors), c(2, 8))
  testthat::expect_true(is.numeric(out$raw_error))
})

testthat::test_that("engine works for gblinear and returns proper structure", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 120, p = 4, seed = 12)
  y <- df[[1]]
  X <- df[, -1, drop = FALSE]

  out <- engine(
    predictors = X, target = y, booster = "gblinear",
    max_depth = 3, eta = 0.2, gamma = 0, min_child_weight = 1,
    subsample = 0.9, colsample_bytree = 0.9,
    lambda = 1, alpha = 0,
    n_windows = 2, patience = 0.2, nrounds = 25
  )

  testthat::expect_true(is.list(out))
  testthat::expect_true(all(c("model", "errors", "raw_error") %in% names(out)))
  testthat::expect_true(inherits(out$model, "xgb.Booster"))
  testthat::expect_true(is.matrix(out$errors))
  testthat::expect_equal(dim(out$errors), c(2, 8))
})

# -------------------------------
# tests/testthat/test-hood.R
# -------------------------------
testthat::test_that("hood runs end-to-end (gbtree, with svd reduction)", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  ts_set <- .make_ts_df(n = 140, p = 4, seed = 21)
  deriv <- rep(0L, ncol(ts_set))

  out <- hood(
    ts_set = ts_set,
    seq_len = 5,
    deriv = deriv,
    norm = FALSE,
    n_dim = 2,              # forces SVD reduction
    ci = 0.8,
    min_set = 30,
    booster = "gbtree",
    max_depth = 3,
    eta = 0.2,
    gamma = 0,
    min_child_weight = 1,
    subsample = 0.9,
    colsample_bytree = 0.9,
    lambda = 1,
    alpha = 0,
    n_windows = 2,
    patience = 0.2,
    nrounds = 20,
    dates = NULL
  )

  .expect_model_bundle(out)
  testthat::expect_equal(length(out$predictions), ncol(ts_set))
  testthat::expect_equal(nrow(out$predictions[[1]]), 5)
  testthat::expect_true(all(c("max_rmse", "max_mae", "max_mdae") %in% colnames(out$joint_error)))
})

testthat::test_that("hood runs end-to-end (gblinear)", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  ts_set <- .make_ts_df(n = 140, p = 3, seed = 22)
  deriv <- rep(0L, ncol(ts_set))

  out <- hood(
    ts_set = ts_set,
    seq_len = 5,
    deriv = deriv,
    norm = FALSE,
    n_dim = ncol(ts_set),
    ci = 0.8,
    min_set = 30,
    booster = "gblinear",
    max_depth = NULL,
    eta = 0.2,
    gamma = NULL,
    min_child_weight = NULL,
    subsample = NULL,
    colsample_bytree = NULL,
    lambda = 5,
    alpha = 2,
    n_windows = 2,
    patience = 0.2,
    nrounds = 20,
    dates = NULL
  )

  .expect_model_bundle(out)
  testthat::expect_equal(length(out$predictions), ncol(ts_set))
})

# -------------------------------
# tests/testthat/test-search.R
# -------------------------------
testthat::test_that("random_search returns history + models (gbtree)", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 130, p = 4, seed = 31)
  deriv <- rep(0L, ncol(df))

  rs <- random_search(
    n_sample = 4,
    data = df,
    booster = "gbtree",
    seq_len = 5,
    deriv = deriv,
    norm = FALSE,
    n_dim = ncol(df),
    ci = 0.8,
    min_set = 30,
    max_depth = 3,
    eta = 0.2,
    gamma = 0,
    min_child_weight = 1,
    subsample = 0.9,
    colsample_bytree = 0.9,
    lambda = NULL,
    alpha = NULL,
    n_windows = 2,
    patience = 0.2,
    nrounds = 15,
    dates = NULL,
    seed = 123
  )

  testthat::expect_true(is.list(rs))
  testthat::expect_true(all(c("history", "models") %in% names(rs)))
  testthat::expect_true(is.data.frame(rs$history))
  testthat::expect_true(is.list(rs$models))
  testthat::expect_true(nrow(rs$history) >= 1)
  testthat::expect_equal(nrow(rs$history), length(rs$models))
  testthat::expect_true("wgt_avg_rank" %in% names(rs$history))
})

testthat::test_that("bayesian_search (base R) returns history + models (gblinear)", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 140, p = 3, seed = 32)
  deriv <- rep(0L, ncol(df))

  bs <- bayesian_search(
    n_sample = 2, n_search = 1,
    booster = "gblinear",
    data = df,
    seq_len = c(4L, 5L),
    deriv = deriv,
    norm = NULL,
    n_dim = c(1L, ncol(df)),
    ci = 0.8,
    min_set = 30,
    max_depth = NULL,
    eta = c(0.1, 0.5),
    gamma = NULL,
    min_child_weight = NULL,
    subsample = NULL,
    colsample_bytree = NULL,
    lambda = c(0, 10),
    alpha = c(0, 10),
    n_windows = 2,
    patience = 0.2,
    nrounds = 15,
    dates = NULL
  )

  testthat::expect_true(is.list(bs))
  testthat::expect_true(all(c("history", "models") %in% names(bs)))
  testthat::expect_true(is.data.frame(bs$history))
  testthat::expect_true(is.list(bs$models))
  testthat::expect_true(nrow(bs$history) >= 1)
  testthat::expect_equal(nrow(bs$history), length(bs$models))
  testthat::expect_true("wgt_avg_rank" %in% names(bs$history))
})

# -------------------------------
# tests/testthat/test-audrex.R
# -------------------------------
testthat::test_that("audrex random search runs end-to-end and imputes NA", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_if_not_installed("imputeTS")
  testthat::skip_on_cran()

  df <- .make_df_with_na(n = 140, seed = 41)

  out <- audrex(
    data = df,
    n_sample = 3,
    n_search = 0,        # random
    smoother = FALSE,
    seq_len = 5,
    diff_threshold = 0.001,
    booster = "gbtree",
    norm = FALSE,
    n_dim = ncol(df),
    ci = 0.8,
    min_set = 30,
    max_depth = 3,
    eta = 0.2,
    gamma = 0,
    min_child_weight = 1,
    subsample = 0.9,
    colsample_bytree = 0.9,
    lambda = NULL,
    alpha = NULL,
    n_windows = 2,
    patience = 0.2,
    nrounds = 15,
    dates = NULL,
    seed = 1
  )

  testthat::expect_true(is.list(out))
  testthat::expect_true(all(c("history", "models", "best_model", "time_log") %in% names(out)))
  testthat::expect_true(is.data.frame(out$history))
  testthat::expect_true(is.list(out$models))
  .expect_model_bundle(out$best_model)
})

testthat::test_that("audrex base-R bayesian_search runs end-to-end", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_if_not_installed("imputeTS")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 150, p = 3, seed = 42)

  out <- audrex(
    data = df,
    n_sample = 2,
    n_search = 1,        # bayesian_search path
    smoother = FALSE,
    seq_len = c(4L, 5L),
    diff_threshold = 0.001,
    booster = "gblinear",
    norm = NULL,
    n_dim = c(1L, ncol(df)),
    ci = 0.8,
    min_set = 30,
    eta = c(0.1, 0.5),
    lambda = c(0, 10),
    alpha = c(0, 10),
    n_windows = 2,
    patience = 0.2,
    nrounds = 15,
    dates = NULL,
    seed = 2
  )

  testthat::expect_true(is.list(out))
  testthat::expect_true(all(c("history", "models", "best_model", "time_log") %in% names(out)))
  testthat::expect_true(nrow(out$history) >= 1)
  .expect_model_bundle(out$best_model)
})

# -------------------------------
# Optional: API compliance smoke test
# -------------------------------
testthat::test_that("xgboost new API usage does not emit watchlist rename warning", {
  testthat::skip_if_not_installed("xgboost")
  testthat::skip_on_cran()

  df <- .make_ts_df(n = 120, p = 4, seed = 55)
  y <- df[[1]]
  X <- df[, -1, drop = FALSE]

  warns <- character(0)
  withCallingHandlers({
    engine(
      predictors = X, target = y, booster = "gbtree",
      max_depth = 3, eta = 0.2, gamma = 0, min_child_weight = 1,
      subsample = 0.9, colsample_bytree = 0.9,
      lambda = 1, alpha = 0,
      n_windows = 2, patience = 0.2, nrounds = 15
    )
  }, warning = function(w) {
    warns <<- c(warns, conditionMessage(w))
    invokeRestart("muffleWarning")
  })

  testthat::expect_false(any(grepl("watchlist", warns, ignore.case = TRUE)))
  testthat::expect_false(any(grepl("renamed to 'evals'", warns, fixed = TRUE)))
})

