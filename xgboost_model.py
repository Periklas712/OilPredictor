from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import optuna

def xgboost_train(splits,best_params):
    results = []

    for i, (X_train, X_test, Y_train, Y_test) in enumerate(splits):

        model = XGBRegressor(   
                **best_params
                )
        
        print(f"\nFold: {i+1}")
        model.fit(X_train, Y_train)

        predict = model.predict(X_test)

        mae = mean_absolute_error(Y_test, predict)
        mse = mean_squared_error(Y_test, predict)
        r2 = r2_score(Y_test, predict)

        print("Scores:")
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

        results.append((mae, mse, r2))

    results = np.array(results)
    print("\n--- Mean scores for XGBOOST ---")
    print(f"MAE: {results[:,0].mean():.4f}")
    print(f"MSE: {results[:,1].mean():.4f}")
    print(f"R²: {results[:,2].mean():.4f}")

    return results

def bayesian_optimization_xgboost(splits, n_trials=50):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800), 
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15),   
            'max_depth': trial.suggest_int('max_depth', 5, 10),  
            
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),  
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),  
            
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'device': 'cuda',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        scores = []
        for X_train, X_test, y_train, y_test in splits:
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=n_trials)
    print(f" Best R² score: {study.best_value:.4f}")
    print(f" Best parameters:")
    
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params, study

# add function to predict with dates given as input 