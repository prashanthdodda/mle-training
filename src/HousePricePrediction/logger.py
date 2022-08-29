# Logging Config
# More on Logging Configuration
# https://docs.python.org/3/library/logging.config.html
# Setting up a config
import logging
import logging.config
import os

# from logging_tree import printout

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    # "formatters": {
    #     "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    # },
    "formatters": {
        "default": {
            "format": "%(asctime)s %(clientip)-15s %(user)-8s %(message)s",  # "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {"format": "%(message)s"},
    },
    "root": {"level": "DEBUG"},
}

# log_file = "logs/HousePricePrediction_log.log"


def configure_logger(
    logger=None,
    cfg=None,
    log_file="logs/HousePricePrediction_log.log",
    console=False,
    log_level="DEBUG",
):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
            logger:
                    Predefined logger object if present. If None a ew logger object will be created from root.
            cfg: dict()
                    Configuration of the logging to be implemented by default
            log_file: str
                    Path to the log file for logs to be stored
            console: bool
                    To include a console handler(logs printing in console)
            log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    dir = os.path.dirname(log_file)
    os.makedirs(dir, exist_ok=True)
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    f = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    if log_file or console:
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            fh.setFormatter(f)
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            sh.setFormatter(f)
            logger.addHandler(sh)

    return logger
