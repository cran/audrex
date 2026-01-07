# audrex 1.0.0

* Added a `NEWS.md` file to track changes to the package.

# audrex 1.0.1

* Added minmax normalization and removed shift feature
* Expanded the available statistics both in predictions and pred_stats
* Added cross-validation through expanding windows
* Added two datasets
* Added link to article in Rpubs

# audrex 2.0.0

* Changed the whole architecture: from one-step function to multi-point models for each sequence
* Added latent dimension reduction with svd
* Added automatic differentiation via recursive F-test for de-trending and removed deriv
* Added Yeo-Johson normalization and removed minmax
* Expanded the available statistics both in predictions and pred_stats

# audrex 3.0.0
What’s new:
* Fully aligned with the new XGBoost R API: deprecated watchlist removed, predictions use the standard predict(model, xgb.DMatrix) method, eliminating future-breaking warnings.
* Bayesian optimization redesigned to use a pure base-R surrogate approach, removing GPfit-related numerical failures while preserving the same function signature and outputs.
* Robust failure handling across optimization and modeling: failed models are safely ignored, edge cases handled explicitly, and searches always return valid results.
* Reduced dependency footprint, relying mainly on base R plus xgboost and imputeTS, simplifying installation and maintenance.
* Cleaner, more stable modeling core (engine, sequencer, hood) with safer cross-validation, early stopping, residual sampling, and base svd() for dimensionality reduction.
* Output and API backward compatibility preserved: existing code using audrex, random_search, or bayesian_search continues to work unchanged.
* Comprehensive test coverage added via testthat, validating all major paths (gbtree, gblinear, random search, Bayesian search, full audrex pipeline).
