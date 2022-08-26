import argparse
import configparser
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV  # , RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from HousePricePrediction import logger

config = configparser.ConfigParser()
config.read("setup.cfg")
log_obj = logging.getLogger(__name__)
logger = logger.configure_logger(
    logger=log_obj,
    log_file=config["params"]["log_file"],
    console=config["params"]["no_console"],
    log_level=config["params"]["log_level"],
)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip",
        "--input_data",
        type=str,
        help="path to the dataset",
        default=config["params"]["processed_data"],
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        help="path to the trained model file",
        default=config["params"]["trained_models"],
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Specify the log level",
        default=config["params"]["log_level"],
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="path to store the logs",
        default=config["params"]["log_file"],
    )

    parser.add_argument(
        "--no-console-log",
        type=str,
        help="Whther to write or not to write the logs to the console",
        default=config["params"]["no_console"],
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_data) or "train.csv" not in os.listdir(
        args.input_data
    ):
        # if not args.input_data:
        parser.error("Please Provide the path to the data for training using  -ip")
    if not args.model_path:
        parser.error(
            "Please Provide the path to store the trained model file using  -mp"
        )
    if args.no_console_log and not args.log_path:
        parser.error(
            "Please Provide the file path to store the logs using --log-path, as you mentioned to not toprint to the console"
        )
    return args


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]


def train(
    train_data, model_path,
):
    """'
    train function takes the train data from the Train object and stores the model in a pickle file

    Attributes
    -----------
        model_path : str
        pickle file path to store the trained model
    """
    logger.info("loading the data..")
    train_data = pd.read_csv(train_data)
    X_train, y_train = (
        train_data.drop(["median_house_value"], axis=1),
        train_data["median_house_value"],
    )

    lin_reg = LinearRegression()
    logger.info("training Linear Regression model")
    lin_reg.fit(X_train, y_train)

    tree_reg = DecisionTreeRegressor(random_state=42)
    logger.info("training decision tree model")
    tree_reg.fit(X_train, y_train)

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    grid_search.best_params_
    # cvres = grid_search.cv_results_
    # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    #     print(np.sqrt(-mean_score), params)

    final_model = grid_search.best_estimator_

    # save models into .pkl files
    lr_path = model_path + "/lr_model.pkl"
    with open(lr_path, "wb") as path:
        pickle.dump(lin_reg, path)

    dt_path = model_path + "/dt_model.pkl"
    with open(dt_path, "wb") as path:
        pickle.dump(tree_reg, path)

    rf_path = model_path + "/rf_model.pkl"
    with open(rf_path, "wb") as path:
        pickle.dump(final_model, path)

    logger.info(f"Trained models stored in path : {model_path}")
    logger.info(f"saved models : {os.listdir(model_path)}")
    return model_path


if __name__ == "__main__":
    args = arg_parser()
    train(args.input_data, args.model_path)

