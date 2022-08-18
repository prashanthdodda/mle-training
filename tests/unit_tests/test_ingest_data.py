import os
import unittest

import pandas as pd
from HousePricePrediction import ingest_data


class Test_ingest_data(unittest.TestCase):
    def test_fetch_housing_data(self):
        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
        HOUSING_PATH = "data/raw/datasets/housing"
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
        ingest_data.fetch_housing_data(housing_path=HOUSING_PATH, housing_url=HOUSING_URL)
        self.assertTrue(
            os.path.exists("data/raw/datasets/housing/housing.csv")
        ), "housing.csv is not present"

    def test_load_data(self):
        data_dir = "data/raw/datasets/housing"
        df = ingest_data.load_housing_data(housing_path=data_dir)
        assert isinstance(df, pd.DataFrame), "expected dataframe"

    def test_split_data(self):
        processed_dir = "data/processed"
        ingest_data.split_data(processed_dir)
        processed_files = os.listdir(processed_dir)
        assert ("train.csv" in processed_files) and ("test.csv" in processed_files)


if __name__ == "__main__":
    unittest.main()
