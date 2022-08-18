from HousePricePrediction import score

# import unittest


def test_train():
    test_data = "data/processed/test.csv"
    model_path = "artifacts/rf_model.pkl"
    rmse = score.find_score(test_data, model_path)
    print(type(rmse))
    assert isinstance(rmse, float)
