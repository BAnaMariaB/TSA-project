# Household Power Consumption: Time Series Forecasting & Anomaly Detection

Self-contained project deliverable. All the analysis lives in a single notebook and needs only the raw data file (included) and the Python packages listed in `requirements.txt`.

## Overview

This project studies four years of household electricity use and answers two practical questions:

1. **Forecasting:** can we predict daily electricity consumption for the next three months, and which model does it best?
2. **Anomaly detection:** which days behave abnormally, and does removing them make the forecasts better?

Working from 2 million raw one-minute meter readings, we build a clean daily series, study its structure (trend, seasonality, stationarity), compare classical and modern forecasting models, flag anomalous days with four different detectors, and then re-forecast after cleaning to measure the payoff.

**Headline result:** the consumption series is dominated by a strong yearly cycle with only a mild weekly pattern. Prophet is the most accurate and most stable forecaster, and the residual diagnostics show exactly why: the classical SARIMA model cannot represent the annual cycle, while Prophet models it directly. Cleaning anomalous days gives both models a small but consistent extra improvement.

## Repository structure

```
deliverables/
├── TSA_Project_Notebook.ipynb   # full analysis, executed with all outputs
├── data/
│   ├── raw/
│   │   └── household_power_consumption.txt   # UCI dataset (1-min readings, Dec 2006 to Nov 2010)
│   └── processed/
│       └── daily_notebook.csv                # cached daily series (created on first run)
├── requirements.txt
└── README.md
```

## Dataset

UCI *Individual Household Electric Power Consumption*: 2,075,259 one-minute measurements from a household in Sceaux, France (2006-12-16 to 2010-11-26). The modeling target is `energy_kwh`, the daily electricity consumption in kWh, aggregated from the minute-level `Global_active_power` readings. The file also carries three sub-meter circuits (kitchen, laundry and fridge, water heater and A/C), which we use for extra context.

## Approach, step by step

1. **Preprocessing.** The raw file is semicolon-separated with missing values marked as `?`. We build a proper datetime index, fill the 25,979 missing values per column by time interpolation, and aggregate the minute readings into a daily series of 1,442 days (about 26 kWh/day on average). Daily granularity keeps everything downstream easy to read and model.

2. **Exploratory analysis and stationarity.** We plot the series with rolling mean and standard deviation, look at day-of-week and monthly profiles, and run the Augmented Dickey-Fuller (ADF) test. The series is stationary in level (p = 0.0040), and the ACF confirms a weekly rhythm (spikes at lags 7, 14, 21). This tells us to use a weekly seasonal period later on.

3. **Decomposition.** We split the series into trend, seasonal, and residual parts using both classical decomposition and the more robust STL method. Both agree that the slow annual cycle carries most of the variation and the weekly pattern is comparatively small, which foreshadows how hard the naive baseline will be to beat.

4. **Forecasting on a 90-day holdout.** Using a chronological split (no shuffling, so no leakage), we compare four approaches: a flat naive forecast, a seasonal naive, SARIMA (with its (p, d, q) order chosen by AIC on the training data only), and Prophet (with light hyperparameter tuning). Models are scored on RMSE, MAE, and MAPE, plus AIC and BIC for SARIMA.

5. **Rolling-origin validation.** Because a single holdout can be misleading, we repeat the whole forecast on four non-overlapping 90-day windows moving backward in time and average the scores. This is where the real ranking of the models emerges.

6. **Residual diagnostics.** We check whether the SARIMA residuals look like white noise, using a time plot, a distribution plot, the residual ACF, a normal Q-Q plot, and the Ljung-Box test. This is the standard sanity check that tells us whether the model captured all the structure it should have.

7. **Sub-metering exploration.** We break the daily total down into the three sub-circuits plus an unmetered remainder, purely from columns already in the raw file, to understand where the energy goes.

8. **Anomaly detection and re-forecasting.** We flag unusual days with four detectors: a z-score (global outliers), standardized SARIMA residuals (contextual outliers), PCA reconstruction error, and Isolation Forest. All four are calibrated on the training window only. We then interpolate the flagged training days, refit both forecasters, and measure whether the forecasts improve on the untouched test window.

## Key results

Forecast accuracy on the 90-day holdout:

| Model (90-day holdout)   | RMSE  | MAE   | MAPE   |
|--------------------------|-------|-------|--------|
| Naive (last value)       | 7.15  | 5.31  | 19.9%  |
| Seasonal naive (m=7)     | 13.35 | 11.57 | 42.0%  |
| SARIMA(0,1,2)x(1,1,1,7)  | 11.48 | 9.62  | 33.4%  |
| Prophet (tuned)          | 7.26  | 5.24  | 22.6%  |

The naive baseline looks competitive on this one holdout, but **rolling-origin validation across four backward folds shows that is an artifact of that specific test window.** Averaged over the four folds, Prophet (mean RMSE 6.25, mean MAPE 20.6%) and SARIMA (mean RMSE 8.32) both beat the naive baseline (mean RMSE 8.44), and Prophet is also the steadiest across folds. So capturing the seasonal and trend structure genuinely helps year-round.

The residual diagnostics explain the ranking. SARIMA's residuals fail the Ljung-Box white-noise test (p = 0.00 at every lag) because a period-7 model cannot represent the dominant annual cycle, while Prophet models that yearly seasonality directly. That is the clearest single reason Prophet comes out ahead.

Anomaly detection flags 45 training days (about 3% of the window) across the four methods. After interpolating those days, both models improve on the untouched test data: SARIMA goes from 11.48 to 10.90 RMSE and Prophet from 7.26 to 7.15.

## Conclusions

- **The series is stationary in level but ruled by a yearly cycle.** Winter consumption is roughly double summer, and that annual swing carries far more variation than the weekly pattern.
- **Prophet is the best forecaster here, and we can say why.** It wins on both RMSE and MAPE in rolling-origin validation and is the most stable across folds. The residual diagnostics pin the cause on SARIMA's inability to model the annual cycle with a weekly seasonal period.
- **Simple baselines deserve respect, but must be tested properly.** The flat naive forecast tied the models on one window and lost across four, which is a reminder that a single holdout can mislead.
- **Cleaning anomalies helps a little, consistently.** Removing the 45 flagged days improved both models without any test-window leakage, showing those days were mildly distorting the fitted structure.

## Limitations and possible extensions

- SARIMA uses only a weekly seasonal period; adding a yearly component (for example via SARIMAX with Fourier terms) would likely close much of the gap with Prophet.
- The AIC grid mixes models with different differencing orders `d`, whose likelihoods are not strictly comparable; a fixed `d` or a unit-root-guided choice would be cleaner.
- Anomaly labels are unsupervised, so the flagged days are best treated as candidates for review rather than confirmed faults.

## How to run


```bash
cd deliverables

# create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# open the notebook
jupyter notebook TSA_Project_Notebook.ipynb
```

Then run everything (Kernel, then Restart & Run All). A full run takes about one to two minutes, and the SARIMA order grid search is the slowest step. The notebook finds `data/raw/household_power_consumption.txt` automatically as long as the folder layout above is kept.
