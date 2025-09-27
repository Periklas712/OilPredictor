import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from periods_costs import covid_periods, war_periods, high_season_periods, revaluation_periods
from periods_costs import cost, olive_oil_price, locations
from sklearn.ensemble import IsolationForest


def load_data(path="data1.xlsx"):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()  
    df["DATE"] = pd.to_datetime(df['DATE'], dayfirst=True)
    df = df.sort_values(by="DATE", ascending=False)
    df['QUANTITY'] = df['QUANTITY'].astype(int).abs()
    df['YEAR'] = df['DATE'].dt.year
    df['MONTH'] = df['DATE'].dt.month
    print("Data loaded and converted to correct types")
    return df

def map_periods(df):
    dates = df['DATE']
    covid_mask = np.zeros(len(df), dtype=bool)
    for start, end in covid_periods:
        covid_mask |= (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))

    war_mask = np.zeros(len(df), dtype=bool)
    for start, end in war_periods:
        war_mask |= (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))

    high_season_mask = np.zeros(len(df), dtype=bool)
    for start, end in high_season_periods:
        high_season_mask |= (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))

    revaluation_mask = np.zeros(len(df), dtype=bool)
    for start, end in revaluation_periods:
        revaluation_mask |= (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))

    df['IS_COVID'] = covid_mask.astype(int)
    df['IS_WAR_PERIOD'] = war_mask.astype(int)
    df['IS_HIGH_SEASON_PERIOD'] = high_season_mask.astype(int)
    df['IS_REVALUATION_PERIOD'] = revaluation_mask.astype(int)
    df['IS_SUMMER'] = df["DATE"].dt.month.isin([6,7,8]).astype(int)
    print("Data mapped to correct periods")
    return df

def map_costs_prices(df):
    sorted_cost = sorted(cost.items(), key=lambda x: x[0][0])
    sorted_olive_price = sorted(olive_oil_price.items(), key=lambda x: x[0][0])
    dates = df['DATE']

    cost_values = np.full(len(df), np.nan)
    for (start, end), value in sorted_cost:
        mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
        cost_values[mask] = value

    olive_prices = np.full(len(df), np.nan)
    for (start, end), value in sorted_olive_price:
        mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
        olive_prices[mask] = value

    df['COST'] = cost_values
    df['OLIVE_OIL_PRICE'] = olive_prices
    df["COST"] = df["COST"].ffill()
    df["OLIVE_OIL_PRICE"] = df["OLIVE_OIL_PRICE"].ffill()
    df["VALUE"] = df["QUANTITY"] * df["SALE_PRICE"]
    df["TAX"] = 13

    if 'LOCATION' in df.columns:
        df["REGION"] = df["LOCATION"].map(locations).fillna(-1).astype(int)
    print("Data mapped to correct cost prices")
    return df

def abs_amounts(df):
    amount_cols = ['COST','SALE_PRICE','VALUE','OLIVE_OIL_PRICE']
    for col in amount_cols:
        df[col] = df[col].abs()
    print("Data converted to positive values")
    return df

def remove_outliers(df,column,factor=1.5): 
    Q1 = df[column].quantile(0.25) 
    Q3 = df[column].quantile(0.75)    
    IQR = Q3 - Q1   
    lower = Q1 - factor * IQR   
    upper = Q3 + factor * IQR  
    print("Removed outliers with iqr method")  
    return df[(df[column] >= lower) & (df[column] <= upper)]

def save_clean_data(df, path="data_filled.xlsx"):
    print("Data saved to excel")
    df.to_excel(path, index=False)
