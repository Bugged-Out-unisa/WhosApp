class ModelUtilities:
    @staticmethod
    def check_prefix_extension(self, name, prefix, extension):
        """Check if the name has the correct prefix and extension."""
        if not name.startswith(prefix):
            name = prefix + name
        if not name.endswith(extension):
            name = name + extension
        return name