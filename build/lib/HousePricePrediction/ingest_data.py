import argparse
import configparser
import logging
import os
import tarfile

# import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer

# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from HousePricePrediction import logger as lg

config = configparser.ConfigParser()
config.read("setup.cfg")
# print("config["params"]["housing_path"] :", config["params"]["housing_path"])


log_obj = logging.getLogger(__name__)
logger = lg.configure_logger(
    logger=log_obj,
    log_file=config["params"]["log_file"],
    console=config["params"]["no_console"],
    log_level=config["params"]["log_level"],
)


def arg_aprser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-op",
        "--outputpath",
        type=str,
        help="path to store the downloaded data",
        default="data/raw/datasets/housing",  # config["params"]["housing_path"],  # "data/raw/datasets/housing",
    )
    parser.add_argument(
        "-d",
        "--train_test_data",
        type=str,
        help="path to store the training and validation datasets",
        default="data/processed",  # config["params"]["processed_data"],  # "data/processed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Specify the log level",
        default="DEBUG",  # config["params"]["log_level"],  # "DEBUG"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="path to store the logs",
        default="logs/HousePricePrediction_log.log",  # config["params"]["log_file"],  # "logs/HousePricePrediction_log.log",
    )
    parser.add_argument(
        "--no-console-log",
        type=str,
        help="Whether to write or not to write the logs to the console",
        default=False,  # config["params"]["no_console"],  # False
    )
    args = parser.parse_args()
    if not args.outputpath:
        parser.error("Please Provide the path to the data using -op")
    # if not args.train_test_data:
    #     parser.error("Please Provide the path to store the processed data using -d")
    if args.no_console_log and not args.log_path:
        parser.error(
            "Please Provide the file path to store the logs using --log-path, as you mentioned to not toprint to the console"
        )

    return args


# class Ingest_data:
#     def __init__(self, args, logger):

#         self.HOUSING_PATH = args.outputpath  # os.path.join("datasets", "housing")
#         self.HOUSING_URL = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"  # config["params"]["housing_url"]
#         self.logger = logger
#         self.logger.info(f"Intiating {self.__class__.__name__}")


def fetch_housing_data(housing_url, housing_path):
    """
    fect_housing_data function in the Ingest_Data takes housing_url and housing_path as the inputs and stores the data.

    Parameters
    ----------
    housing_url: url
            url to download the data from.
    housing_path: str
            path to store the data that was downloaded from the housing_url.

    Returns
    -------
        Doesn't return anything. It downloads the data and stores it
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logger.info(f"fetching housing data from {tgz_path}")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """
    The function load_housing_data in Ingest_data takes housing_path as the input and returns the dataframe

    Parameters
    ----------
    housing_path: str
            path to read the data from

    Returns
    -------
    df

    """
    logger.info("loading housing data")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# def income_cat_proportions(self, data):
#     """
#     this function income_cat_proportions take the data and makes teh income_cat into proportions
#     Parameters
#     ----------
#             data : dataframe
#     Returns
#     ----------
#     income category proportions

#     """
#     return data["income_cat"].value_counts() / len(data)


def split_data(train_test_data_path):
    """
    split_data function splits the data into train and test datsets and stores them in the csv fomat in the directory that was provoided in the arguments.
    default directory to store the train adn test data is data/processed

    Attributes
    ----------
    train_test_data_path : str
            path to store the train and test datasets.

    """
    housing_path = config["params"]["housing_path"]
    # train_test_data_path = config['params']['processed_data']

    logger.info("Splitting the data into train and test")
    # fetch the data from the URL
    fetch_housing_data(
        housing_url=config["params"]["housing_url"], housing_path=housing_path
    )

    # load the data
    housing = load_housing_data(housing_path)

    # adding new columns
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    # split X and y
    X = housing.drop(["median_house_value"], axis=1)
    y = housing["median_house_value"]

    # splitting the data
    # split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # for train_index, test_index in split.split(X, y):
    #     X_train, y_train = X.loc[train_index], y.loc[train_index]
    # X_test, y_test = X.loc[test_index], y.loc[test_index]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Imputing missing values in numerical variables
    imputer = SimpleImputer(strategy="median")
    train_num = X_train.drop(["ocean_proximity"], axis=1)
    cols = train_num.columns
    imputer.fit(train_num)
    train_num = imputer.transform(train_num)

    test_num = X_test.drop(["ocean_proximity"], axis=1)
    test_num = imputer.transform(test_num)

    # combining numerical and categorical variables
    train_num = pd.DataFrame(train_num, columns=cols, index=X_train.index)
    test_num = pd.DataFrame(test_num, columns=cols, index=X_test.index)

    # coverting the categorical variables into one hot encoding
    train_cat = pd.get_dummies(X_train["ocean_proximity"])
    test_cat = pd.get_dummies(X_test["ocean_proximity"])

    # combine numerical and categorical variables
    train_combined = train_num.join(train_cat)
    test_combined = test_num.join(test_cat)

    # create train and test data
    train_data = train_combined.join(y_train)
    test_data = test_combined.join(y_test)
    # test_data = (test_combined, y_test)

    train_data.to_csv(f"{train_test_data_path}/train.csv")
    test_data.to_csv(f"{train_test_data_path}/test.csv")

    logger.debug(
        f"splitted data is stored in the path {os.listdir(train_test_data_path)}"
    )
    return train_test_data_path


if __name__ == "__main__":
    args = arg_aprser()
    split_data(args.train_test_data)
