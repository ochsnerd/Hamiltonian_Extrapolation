import os
import pickle

from typing import Callable, Any, Union
from pathlib import Path


def load_or_compute(func: Callable[[], Any],
                    name: str,
                    directory: Union[str, Path] = "") -> Any:
    """Attempt to load result of func from disc, not successful compute it.

    func is a function that takes no arguments and returns a pickleable
    object. Us a lambda to transform a function taking arguments into
    a callable taking no arguments.

    >>> load_or_compute(lambda: my_func(many, arguments, here), "name")

    func is only evaluated if no file is found.

    The file is looked for in the directory indicated by the
    environment-variable DATA_DIR. If no such variable exists,
    $HOME/Documents is used instead.
    """
    if not directory:
        directory = get_data_dir()
    path = os.path.join(directory, name)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Computing", name)
        res = func()
        with open(path, 'wb') as f:
            pickle.dump(res, f)
        return res


def get_data_dir(env_var_name: str = 'DATA_DIR') -> str:
    try:
        data_dir = os.environ[env_var_name]
        if not data_dir:
            raise RuntimeError
        return data_dir
    except (KeyError, RuntimeError):
        print(f"{env_var_name} not set, using ~/Documents/")
        return os.path.join(os.environ['HOME'], "Documents")
