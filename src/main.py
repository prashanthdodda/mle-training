import os

import mlflow
import mlflow.sklearn

from HousePricePrediction import ingest_data, logger, score, train


def get_data():
    ingest_data.fetch_housing_data
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw/datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    processed_dir = "data/processed"
    ingest_data.load_housing_data(housing_path=HOUSING_PATH)
    train_test_path = ingest_data.split_data(
        processed_dir, housing_path=HOUSING_PATH, housing_url=HOUSING_URL
    )
    train_data_path = train_test_path + "/train.csv"
    test_data_path = train_test_path + "/test.csv"
    return train_data_path, test_data_path


def train_model(train_data_path, test_data_path, model_dir="artifacts"):
    models = train.train(train_data_path, model_dir)
    # model_path = os.path.join(model_dir, os.listdir(model_dir)[2])
    return models


def get_score(test_data_path, model_path):
    res = score.find_score(test_data_path, model_path)
    return res


def run():
    train_data_path, test_data_path = get_data()
    model_path = train_model(train_data_path, test_data_path)
    for model in model_path:
        res = get_score(test_data_path, model)
    return res


# if __name__ == "__main__":
#     run()

# Checking if the script is executed directly
if __name__ == "__main__":
    # get train and test data
    train_data_path, test_data_path = get_data()

    trained_models = train.train(train_data_path)
    experiment_id = mlflow.create_experiment("experiment1")
    # Starting a tracking run
    with mlflow.start_run(run_name="PARENT_RUN", experiment_id=experiment_id):
        # For each model, running a child run
        for model in trained_models:
            for metric in ["rmse", "mae"]:
                with mlflow.start_run(
                    run_name="CHILD_RUN", nested=True, experiment_id=experiment_id
                ):
                    res = score.find_score(test_data_path, model, metric)
                    # Logging the test performance of the current model
                    mlflow.log_metric(key=metric, value=res)

                    # Saving the model as an artifact
                    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
