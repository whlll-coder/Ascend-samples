"""log.py"""
import logging


def get_logger(name='root'):
    """return logger"""
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(handler)
    return _logger


logger = get_logger('root')
