# Changelog

## v0.2 – Improved Model (Ridge)
### Changed
- Replaced `LinearRegression` with `Ridge(alpha=1.0)` to add L2 regularization.
- Retrained the model; **RMSE improved from 53.85 → 53.78 (Δ −0.07)**.
- Rationale: L2 regularization slightly reduces variance on correlated features in the diabetes dataset.

## v0.1 – Baseline
- `StandardScaler + LinearRegression`
- RMSE: 53.85
