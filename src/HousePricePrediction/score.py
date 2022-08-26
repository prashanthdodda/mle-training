import argparse
import configparser
import logging
import os
import pickle

import mlflow
import numpy as np
import pandas as pd

# from logging_tree import printout
from sklearn.metrics import mean_absolute_error, mean_squared_error

from HousePricePrediction import logger

# import pickle


# import pandas as pd

config = configparser.ConfigParser()
config.read("setup.cfg")

log_obj = logging.getLogger(__name__)
logger = logger.configure_logger(logger=log_obj)
# logger = logger.configure_logger(
#     log_file="logs/HousePricePrediction_log.log", logger=log_obj
# )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="path to the trained model",
        default=config["params"]["trained_models"] + "/rf_model.pkl",
    )
    parser.add_argument(
        "-td",
        "--test-data",
        type=str,
        help="path to test data",
        default="data/processed",
    )
    # parser.add_argument(
    #     "--log-level",
    #     type=str,
    #     help="Specify the log level",
    #     default=config["params"]["log_level"],
    # )
    # parser.add_argument(
    #     "--log-path",
    #     type=str,
    #     help="path to store the logs",
    #     default=config["params"]["log_file"],
    # )
    # parser.add_argument(
    #     "--no-console-log",
    #     type=str,
    #     help="Whther to write or not to write the logs to the console",
    #     default=config["params"]["no_console"],
    # )
    parser.add_argument(
        "--metric",
        type=str,
        help="Please specify one of the following : 'rmse', 'mae'",
        default=config["params"]["metric"],
    )
    args = parser.parse_args()
    if (
        not args.model_path
        or not os.path.exists(args.model_path)
        or os.path.splitext(args.model_path)[1] != ".pkl"
    ):
        parser.error("Please Provide the path to the 'trained model' using  -mp")

    if not os.path.exists(args.test_data) or "test.csv" not in os.listdir(
        args.test_data
    ):
        parser.error("Please Provide the path to the data for test using  -td")

    # if args.no_console_log and not args.log_path:
    #     parser.error(
    #         "Please Provide the file path to store the logs using --log-path, as you mentioned to not toprint to the console"
    #     )

    return args


# class Score:
#     def __init__(self, args, logger) -> None:
#         """
#         Intializing the test data and house price preditcion model
#         """

#         # load train test data
#         with open(args.model_path, "rb") as f:
#             model = pickle.load(f)
#         self.test_data = pd.read_csv(os.path.join(args.test_data, "test.csv"))
#         self.final_model = model
#         self.logger = logger
#         self.logger.info(f"Intiating {self.__class__.__name__}")


def find_score(test_data, model_path, metric="rmse"):
    """
    Calculates the model score

    Parameters
    ----------
    model: obj
        Fitted Estimator
    X_test: pandas.DataFrame
        features of testing dataset.
    y_test: pandas.DataFrame
        target variable in testing dataset

    Returns
    -------
    tuple
        float: rmse score
        float: mae score
    """
    test_data = pd.read_csv(test_data)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X_test_prepared, y_test = (
        test_data.drop(["median_house_value"], axis=1),
        test_data["median_house_value"],
    )
    final_predictions = model.predict(X_test_prepared)
    res = None
    with mlflow.start_run(run_name="Score"):
        if metric == "rmse":
            final_mse = mean_squared_error(y_test, final_predictions)
            res = np.sqrt(final_mse)
        elif metric == "mae":
            res = mean_absolute_error(y_test, final_predictions)
        else:
            raise Exception("Metric should be one of the following : 'rmse','mae'")
        mlflow.log_metric(key=metric, value=res)

    logger.info(f"FINAL RMSE : {res}")
    return res


if __name__ == "__main__":
    args = arg_parser()
    find_score(args.test_data + "/test.csv", args.model_path, args.metric)
