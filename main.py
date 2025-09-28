from data_preprocessing import load_data, map_periods, map_costs_prices, abs_amounts, save_clean_data ,remove_outliers,add_calendar_features
from feature_engineering import add_lags, standard_normalize, minmax_normalize, extract_feature_importance,time_series_split,expanding_window_split
from visualization import plot_sales_per_year, plot_price_change, simple_heatmap, simple_line_plot,plot_trends_with_periods
from periods_costs import train_end_dates
from xgboost_model import xgboost_train ,bayesian_optimization_xgboost
from model_hyperparameters import xgboost_hyperparameters


# LIGHTGBM REGRESSOR - XGBOOST REGRESSOR ok  - CATBOOST REGRESSOR # RANDOM FOREST REGRESSOR - SARIMA - PHROPHET - LSTM - TFT - N-BEATS

# Load & preprocess
df = load_data("data1.xlsx")
df = abs_amounts(df)
df = map_periods(df)
df = map_costs_prices(df)
df = add_lags(df)
df = add_calendar_features(df)
df = remove_outliers(df,'QUANTITY',2.0)
# save_clean_data(df, "data_filled.xlsx")

# Feature importance
# extract_feature_importance(df)

# Visualizations (uncomment if needed)
# plot_sales_per_year(df)
# plot_price_change(df)
# simple_heatmap(df)
# simple_line_plot(df)
# plot_trends_with_periods(df)

# # Normalize
# # df_std = standard_normalize(df.copy(), columns_to_normalize)
# # df_minmax = minmax_normalize(df.copy(), columns_to_normalize)
     
splits_time_series = time_series_split(df)
split_expanding_window = expanding_window_split(df,train_end_dates)

ans = input("Train with time series split: 1 | Train with expanding window split: 2 :")
ans2 = input("Optimize (1) | Default hyperparameters (2) :")

if ans == "1":  
    if ans2 == "1":  
        best_params, study = bayesian_optimization_xgboost(splits_time_series, n_trials=20)
    else:  
        best_params = xgboost_hyperparameters
    results = xgboost_train(splits_time_series, best_params)

elif ans == "2":  
    if ans2 == "1":  
        best_params, study = bayesian_optimization_xgboost(split_expanding_window, n_trials=20)
    else:  
        best_params = xgboost_hyperparameters
    results = xgboost_train(split_expanding_window, best_params)