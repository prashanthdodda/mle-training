#MLE training
# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.


Modules:
Module ingest_data:
    to fetch and load the hosuing data and to split the data and train and test sets.
    this stores the train and test csv's in data/processed
Module train:
    taking the train data here we build different model to predict the house price.
    used :
        - Linear regression
        - Decision Tree
        - Random Forest
    returns:
        - trained model paths
Module score:
    inputs:
        - takes testdata path and the model path as the inputs. metric ['rmse'/'mae'] optional, default: 'rmse'
    returns:
        - the evaluated metric value

# to install the package HousePricePrediction
- create a virtual environment using the env.yml file
- now activate the environment
    ex:  conda activate mle-dev
- install the .whl file in the folder dist
    pip install dist/HousePricePrediction-0.4-py3-none-any.whl
- to test the package, run python
    import HousePricePrediction
or
- python -v tests/installation_tests/test_installation.py

Note:
    This Should run without any errors if the package is installed properly
