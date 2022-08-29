import os

from HousePricePrediction import ingest_data, logger, score, train


def get_data():
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw/datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    ingest_data.load_housing_data(housing_path=HOUSING_PATH)
    train_data, test_data = ingest_data.split_data(processed_dir)
    # train_data = train_test_path + "/train.csv"
    # test_data = train_test_path + "/test.csv"
    return train_data, test_data


def train_model(train_data, model_dir="artifacts"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = train.train(train_data, model_dir)
    model_path = os.path.join(model_dir, os.listdir(model_dir)[2])

    return model_path


def get_score(test_data, model_path):
    res = score.find_score(test_data, model_path)
    return res


def run():
    train_data_path, test_data_path = get_data()
    model_path = train_model(train_data_path)
    res = get_score(test_data_path, model_path)
    assert isinstance(res, float), "Expected float value"
    print("Model Used for evaluation : ", model_path)
    print("RMSE :", res)
    return res


if __name__ == "__main__":
    run()
