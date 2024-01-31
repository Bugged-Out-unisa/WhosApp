import os


def validation(name, doc=None, *validate_function):
    def decorator(decore_class):
        private_name = "__" + name

        def get_param(self):
            return getattr(self, private_name)

        def set_param(self, value):
            for function in validate_function:
                value = function(value)
            setattr(self, private_name, value)

        setattr(decore_class, private_name, property(get_param, set_param, doc=doc))
        return decore_class

    return decorator


def ensure_valid_file_extension(extension: str):
    """
        Decoratore che controlla se l'estensione del file è corretta.
        Se non lo è, la aggiunge.
    """

    def decorator(name, value):
        if not value.endswith(extension):
            value += extension
        return value

    return decorator


def ensure_file_exists(base_path, subdir, allow_none=False):
    """
        Decoratore che controlla se il file esiste.
        Se non esiste, lancia un'eccezione.
    """

    def decorator(name, value):
        if value is None and allow_none:
            return None

        full_path = os.path.join(base_path, subdir, value)
        if os.path.exists(full_path):
            return value
        else:
            raise FileNotFoundError(f"File {value} not found in {subdir} directory.")

    return decorator


def ensure_valid_type(param_type, allow_none=False):
    """
        Decoratore che controlla il tipo di un parametro.
        Se non è del tipo corretto, lancia un'eccezione.
    """

    def decorator(name, value):
        if value is None and allow_none:
            return None
        if isinstance(value, param_type):
            return value
        else:
            raise TypeError(f"Parameter {name} must be {param_type} type.")

    return decorator
