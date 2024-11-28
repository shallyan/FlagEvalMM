import logging

_DEFAULT_LOGGER_NAME = "flagevalmm"

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(levelname)s - %(message)s, # don't show filename and lineno
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)
