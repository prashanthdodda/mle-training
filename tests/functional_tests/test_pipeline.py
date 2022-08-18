import os
import unittest

from HousePricePrediction import ingest_data, score, train


def test_pipleine():
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw/datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    processed_dir = "data/processed"
    ingest_data.load_housing_data(housing_path=HOUSING_PATH)
    train_test_path = ingest_data.split_data(processed_dir)
    train_data = train_test_path + "/train.csv"
    test_data = train_test_path + "/test.csv"
    model_dir = "artifacts"
    model_dir = train.train(train_data, model_dir)
    model_path = os.path.join(model_dir, os.listdir(model_dir)[2])
    res = score.find_score(test_data, model_path)
    assert isinstance(res, float), "Expected float value"
    return train_test_path

