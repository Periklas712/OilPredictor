from data_preprocessing import load_data, map_periods, map_costs_prices, abs_amounts, save_clean_data ,remove_outliers
from feature_engineering import add_lags, standard_normalize, minmax_normalize, extract_feature_importance,time_series_split
from visualization import plot_sales_per_year, plot_price_change, simple_heatmap, simple_line_plot,plot_trends_with_periods
from periods_costs import columns_to_normalize
import pandas as pd



# LIGHTGBM REGRESSOR - XGBOOST REGRESSOR - CATBOOST REGRESSOR # RANDOM FOREST REGRESSOR - SARIMA - PHROPHET - LSTM - TFT - N-BEATS

# Load & preprocess
df = load_data("data1.xlsx")
df = abs_amounts(df)
df = map_periods(df)
df = map_costs_prices(df)
df = add_lags(df)
df = remove_outliers(df,'QUANTITY',1.5)
save_clean_data(df, "data_filled.xlsx")

splits = time_series_split(df)

# Normalize
# df_std = standard_normalize(df.copy(), columns_to_normalize)
# df_minmax = minmax_normalize(df.copy(), columns_to_normalize)

# Feature importance
# extract_feature_importance(df)

# Visualizations (uncomment if needed)
# plot_sales_per_year(df)
# plot_price_change(df)
# simple_heatmap(df)
# simple_line_plot(df)
# plot_trends_with_periods(df)