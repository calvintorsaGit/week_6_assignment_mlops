import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient

# Initialize tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def get_or_create_experiment(name):
    experiment = client.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    else:
        return client.create_experiment(name)

def train_model(experiment_name, model_class, params, X_train, y_train, X_val, y_val):
    experiment_id = get_or_create_experiment(experiment_name)
    
    with mlflow.start_run(experiment_id=experiment_id):
        # Log basic info
        mlflow.set_tag("model_type", str(model_class.__name__))
        mlflow.log_params(params)
        
        # Train model
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # Log artifacts (the model)
        with open("model.bin", "wb") as f_out:
            pickle.dump(model, f_out)
        mlflow.log_artifact("model.bin")
        
        # Evaluate
        y_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        print(f"Finished {experiment_name} with RMSE: {rmse:.4f}")
        return model

def main():
    # Load prepared data
    print("Loading data...")
    df_train = pd.read_parquet('train.parquet')
    
    # Simple split for local validation (e.g., use last 10% for internal check)
    # We will use the proper test.parquet (March) in Task 2.
    # For now, let's split df_train slightly to log meaningful metrics.
    train_size = int(0.8 * len(df_train))
    df_train_sub = df_train[:train_size]
    df_val_sub = df_train[train_size:]

    categorical = ['PULocationID', 'DOLocationID']
    dv = DictVectorizer()

    train_dicts = df_train_sub[categorical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val_sub[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = df_train_sub['duration'].values
    y_val = df_val_sub['duration'].values

    # Step 1: Linear Regression
    lr_params = {}
    train_model("GreenTaxi-Linear", LinearRegression, lr_params, X_train, y_train, X_val, y_val)

    # Step 2: Lasso
    lasso_params = {'alpha': 0.1}
    train_model("GreenTaxi-Lasso", Lasso, lasso_params, X_train, y_train, X_val, y_val)

    # Step 3: Random Forest (Limited depth and estimators for speed)
    rf_params = {'max_depth': 10, 'n_estimators': 10, 'random_state': 42}
    train_model("GreenTaxi-RandomForest", RandomForestRegressor, rf_params, X_train, y_train, X_val, y_val)

    # Save vectorizer for future inference
    with open("dict_vectorizer.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    
    print("\nExperiments complete. View results by running 'mlflow ui --backend-store-uri sqlite:///mlflow.db'")

if __name__ == "__main__":
    main()
