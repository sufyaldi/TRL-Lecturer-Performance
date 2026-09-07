import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

def run_baselines(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(kernel='rbf')
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")
        
    return results

if __name__ == "__main__":
    print("Baseline models (Aggregate Features)")
    # Example placeholder execution
    # run_baselines(X_train, X_test, y_train, y_test)
