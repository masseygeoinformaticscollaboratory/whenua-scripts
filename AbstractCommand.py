import os
import pathlib


class AbstractCommand:
    def __init__(self, _file_):
        current_dir = os.path.dirname(os.path.abspath(_file_))
        script_name = os.path.split(_file_)[1][0:-3]
        self.cache_dir = os.path.join(current_dir, 'cache', script_name)
        pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, cache_name):
        return os.path.join(self.cache_dir, cache_name)
