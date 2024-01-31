import os
from functools import wraps


def check_extension_file(extension_file: str):
    """
        Decoratore che controlla se l'estensione del file è corretta.
        Se non lo è, aggiunge l'estensione corretta.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            filename = function(*args, **kwargs)
            if not filename.endswith(extension_file):
                filename += extension_file
            return filename

        return wrapper
    return decorator


def check_file_exists(base_path, subdir, allow_none = False):
    """
        Decoratore che controlla se il file esiste.
        Se non esiste, lancia un'eccezione.
    """
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            filename = function(*args, **kwargs)
            if filename is None and allow_none:
                return None

            full_path = os.path.join(base_path, subdir, filename)
            if os.path.exists(full_path):
                return filename
            else:
                raise FileNotFoundError(f"File {filename} not found in {subdir} directory.")

        return wrapper
    return decorator


def check_type_param(param_name, param_type, allow_none = False):
    """
        Decoratore che controlla il tipo di un parametro.
        Se non è del tipo corretto, lancia un'eccezione.
    """
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            param = function(*args)
            if param is None and allow_none:
                return None

            if isinstance(param, param_type):
                return function(*args, **kwargs)
            else:
                raise TypeError(f"{param_name} must be a {param_type}")

        return wrapper
    return decorator
