import os
import unittest

from HousePricePrediction import ingest_data, score, train


def test_pipleine():
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw/datasets/housing"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    processed_dir = "data/processed"
    ingest_data.load_housing_data(housing_path=HOUSING_PATH)
    train_test_path = ingest_data.split_data(
        processed_dir, housing_path=HOUSING_PATH, housing_url=HOUSING_URL
    )
    train_data = train_test_path + "/train.csv"
    test_data = train_test_path + "/test.csv"
    model_dir = "artifacts"
    models = train.train(train_data, model_dir)
    res = score.find_score(test_data, models[2])
    assert isinstance(res, float), "Expected float value"
    return train_test_path

