import os
from typing import Optional
from enhancer.utils.config import Files


def check_files(root_dir: str, files: Files):

    path_variables = [
        member_var
        for member_var in dir(files)
        if not member_var.startswith("__")
    ]
    for variable in path_variables:
        path = getattr(files, variable)
        if not os.path.isdir(os.path.join(root_dir, path)):
            raise ValueError(f"Invalid {path}, is not a directory")

    return files, root_dir


def merge_dict(default_dict: dict, custom: Optional[dict] = None):

    params = dict(default_dict)
    if custom:
        params.update(custom)
    return params
