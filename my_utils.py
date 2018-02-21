import json
import logging
import tensorflow as tf

from time import time, sleep

DEFAULT_LOGGER = None


class Timer:
    def __enter__(self):
        self._tic = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.eclipsed = time() - self._tic


def sleep_secs(n_secss):
    sleep(n_secss)


def get_default_logger():
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = logging.getLogger('ALL')
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s'))

        DEFAULT_LOGGER.setLevel(logging.DEBUG)
        DEFAULT_LOGGER.addHandler(handler)
    return DEFAULT_LOGGER


def load_config(path=None, **kwargs):
    path = 'config.json' if path is None else path
    with open(path, 'r') as f:
        config = json.load(f)
        config.update(kwargs)
    return config


def json_out(inp):
    return json.dumps(inp, indent=2)


def delete_if_exists(path):
    if tf.gfile.Exists(path):
        tf.gfile.DeleteRecursively(path)
