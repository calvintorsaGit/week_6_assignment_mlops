import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error
import mlflow
from mlflow.tracking import MlflowClient

# Initialize tracking URI
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

REGISTERED_MODEL_NAME = "GreenTaxi-Regression-Model"

def get_best_run(experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"]
    )
    return runs[0] if runs else None

def main():
    # 1. Load March data
    print("Loading March data and vectorizer...")
    df_test = pd.read_parquet('test.parquet')
    with open("dict_vectorizer.bin", "rb") as f_in:
        dv = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    test_dicts = df_test[categorical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)
    y_test = df_test['duration'].values

    # 2. Evaluate best runs from each experiment
    experiments = ["GreenTaxi-Linear", "GreenTaxi-Lasso", "GreenTaxi-RandomForest"]
    model_results = []

    for exp_name in experiments:
        best_run = get_best_run(exp_name)
        if not best_run:
            continue
        
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        # Load and test
        model = mlflow.pyfunc.load_model(model_uri)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        
        # Log the March RMSE back to the original run so it's visible in the UI
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("march_rmse", rmse)
        
        print(f"Model from {exp_name} (Run {run_id}) - March RMSE: {rmse:.4f}")
        
        # 3. Register the model version
        result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        version = result.version
        
        model_results.append({
            'exp': exp_name,
            'run_id': run_id,
            'version': version,
            'rmse': rmse
        })

    # 4. Sort results find best
    model_results.sort(key=lambda x: x['rmse'])
    
    # 5. Transition stages in code
    for i, res in enumerate(model_results):
        # Best model (index 0) to Production, others to Staging
        new_stage = "Production" if i == 0 else "Staging"
        
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=res['version'],
            stage=new_stage,
            archive_existing_versions=False
        )
        print(f"Version {res['version']} ({res['exp']}) transitioned to {new_stage}")

    print("\nModel registry updated successfully!")

if __name__ == "__main__":
    main()
