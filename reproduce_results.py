import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error
import mlflow

# Initialize tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# The name of our model in the Registry
REGISTERED_MODEL_NAME = "GreenTaxi-Regression-Model"
# Loading from the 'Production' stage
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/Production"

def main():
    # 1. Load the model from MLflow Registry
    print(f"Loading Production model from {MODEL_URI}...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # 2. Load the test dataset (March 2021)
    print("Loading test dataset and vectorizer...")
    df_test = pd.read_parquet('test.parquet')
    with open("dict_vectorizer.bin", "rb") as f_in:
        dv = pickle.load(f_in)

    # 3. Prepare data
    categorical = ['PULocationID', 'DOLocationID']
    test_dicts = df_test[categorical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_test = df_test['duration'].values

    # 4. Run inference again
    print("Running inference...")
    y_pred = model.predict(X_test)

    # 5. Calculate evaluation metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print("-" * 30)
    print(f"Reproduction RMSE: {rmse:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
