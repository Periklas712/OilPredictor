import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def add_lags(df):
    daily = df.groupby('DATE', as_index=False)['QUANTITY'].sum().sort_values('DATE')
    daily['LAG_1'] = daily['QUANTITY'].shift(1)
    daily['LAG_7'] = daily['QUANTITY'].shift(7)
    daily['ROLL_7'] = daily['QUANTITY'].rolling(window=7, min_periods=1).mean()
    daily['ROLL_14'] = daily['QUANTITY'].rolling(window=14, min_periods=1).mean()
    df = df.merge(daily[['DATE','LAG_1','LAG_7','ROLL_7','ROLL_14']], on='DATE', how='left')
    print("Created lags and rolls for data")
    return df

def standard_normalize(df, columns):
    scaler = StandardScaler()
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])
    print("Data normalized with standard scaler")
    return df

def minmax_normalize(df, columns):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])
    print("Data normalized with min max scaler")
    return df

def extract_feature_importance(df):
    feature_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    exclude = ['QUANTITY','VALUE']
    feature_cols = [c for c in feature_cols if c not in exclude]
    X = df[feature_cols]
    y = df['QUANTITY']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10,6))
    plt.barh(fi_df['feature'], fi_df['importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.show()
    print("Created feature importance plot")


def time_series_split(df):
    df = df.sort_values("DATE")

    features = df.drop(columns=["QUANTITY"])
    prediction = df["QUANTITY"]
    
    time_series_split = TimeSeriesSplit(n_splits=5)

    splits = []
    for train_idx , test_idx in time_series_split.split(features):
        X_train , X_test = features.iloc[train_idx] , features.iloc[test_idx]
        Y_train , Y_test = prediction.iloc[train_idx] , prediction.iloc[test_idx]
        splits.append((X_train,X_test,Y_train,Y_test))
    print("Data splited with times series split")
    return splits


