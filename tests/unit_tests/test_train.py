import os

from HousePricePrediction import train

# import unittest


def test_train():
    input_data = "data/processed/train.csv"
    model_path = "artifacts"
    train.train(input_data, model_path)
    model_dir = "artifacts"
    artifacts_files = os.listdir(model_dir)
    assert "rf_model.pkl" in artifacts_files
