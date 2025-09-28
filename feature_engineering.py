import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# This function is used to create the lags and rolls for 1 , 14 , 30 days
# Params : the dataframe 
# Returns : new dataframe with the new values 
def add_lags(df):
    daily = df.groupby('DATE', as_index=False)['QUANTITY'].sum().sort_values('DATE')
    
    for lag in [7, 14, 30]:
        daily[f'LAG_{lag}'] = daily['QUANTITY'].shift(lag)
    
    for window in [7, 14, 30]:
        daily[f'ROLL_MEAN_{window}'] = daily['QUANTITY'].shift(1).rolling(window=window, min_periods=1).mean()
        daily[f'ROLL_STD_{window}'] = daily['QUANTITY'].shift(1).rolling(window=window, min_periods=1).std()
    
    lag_columns = ['DATE'] + [col for col in daily.columns if col.startswith(('LAG_', 'ROLL_'))]    
    df = df.merge(daily[lag_columns], on='DATE', how='left')
    
    print("Created lags and rolling features for TOTAL daily sales prediction")
    return df

# This function is used to normalize the data with the standard sklearn scaler
# Params : the dataframe 
# Returns : new dataframe with the normalized values
def standard_normalize(df, columns):
    scaler = StandardScaler()
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])
    print("Data normalized with standard scaler")
    return df

# This function is used to normalize the data with the min max sklearn scaler
# Params : the dataframe 
# Returns : new dataframe with the normalized values
def minmax_normalize(df, columns):
    scaler = MinMaxScaler()
    for col in columns:
        df[col] = scaler.fit_transform(df[[col]])
    print("Data normalized with min max scaler")
    return df

# This function is used to exctract the importance of every feature on the dataframe using a random forest regressor 
# Also visualizes in a plot every feature 
# Params : the dataframe
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

# This function is used to split the datase into training and test set using the time series split into folds exlcuding the non numerical columns
# Params : the dataframe 
# Returns : every split for every fold 
def time_series_split(df):
    df = df.sort_values("DATE")

    features = df.drop(columns=["QUANTITY","LOCATION","DATE"])
    prediction = df["QUANTITY"]
    
    time_series_split = TimeSeriesSplit(n_splits=4)

    splits = []
    for train_idx , test_idx in time_series_split.split(features):
        X_train , X_test = features.iloc[train_idx] , features.iloc[test_idx]
        Y_train , Y_test = prediction.iloc[train_idx] , prediction.iloc[test_idx]
        splits.append((X_train,X_test,Y_train,Y_test))
    print("Data splited with times series split")
    return splits

# This function is used to split the datase into training and test set using using an expanding window split exlcuding the non numerical columns
# We seperate every fold based on 365 days and every new fold another year is added to the test 
# Params : the dataframe , the training days (each year) , the period for every fold 
# Returns : every split for every fold 
def expanding_window_split(df, train_end_dates, test_period_days=365):
    """
    train_end_dates: λίστα με ημερομηνίες που τελειώνει κάθε training period
    test_period_days: πόσες μέρες θα περιέχει το test set μετά το train_end
    """
    df = df.sort_values("DATE")
    features = df.drop(columns=["QUANTITY", "LOCATION", "DATE"])
    target = df["QUANTITY"]

    splits = []

    for train_end in train_end_dates:
        train_idx = df[df['DATE'] <= train_end].index
        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.Timedelta(days=test_period_days)
        test_idx = df[(df['DATE'] >= test_start) & (df['DATE'] <= test_end)].index

        if len(test_idx) == 0:
            continue

        X_train, X_test = features.loc[train_idx], features.loc[test_idx]
        y_train, y_test = target.loc[train_idx], target.loc[test_idx]

        splits.append((X_train, X_test, y_train, y_test))

    print(f"Data split with expanding window based on dates, total folds: {len(splits)}")
    return splits




