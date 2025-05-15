import calendar
import os
import time


class ModelUtilities:
    @staticmethod
    def check_output_model_name(prefix: str, extension: str, name: str = None) -> str:
        """Check if the model name is valid."""
        if name:
            return ModelUtilities.check_prefix_extension(name, prefix, extension)
        else:
            return prefix + str(calendar.timegm(time.gmtime())) + extension

    @staticmethod
    def check_prefix_extension(name: str, prefix: str, extension: str) -> str:
        """Check if the name has the correct prefix and extension."""
        if not name.startswith(prefix):
            name = prefix + name
        if not name.endswith(extension):
            name = name + extension
        return name
    
    @staticmethod
    def check_duplicate_model_name(name: str, retrain: bool, path: str = "./"):
        """Check if the model name already exists."""

        if os.path.exists(os.path.join(path, name)) and not retrain:
            raise ValueError(f"Model with name {name} already exists. Allow retraining to overrite.")
        
    @staticmethod
    def check_not_none(variable, name: str):

        """Check if the variable is not None."""
        if variable is None:
            raise ValueError(f"Variable '{name}' cannot be None.")
        
        return variable
    
    @staticmethod
    def check_path(path: str) -> str:
        """Check if the path exists."""
        if path is None:
            raise ValueError("Path cannot be None.")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path