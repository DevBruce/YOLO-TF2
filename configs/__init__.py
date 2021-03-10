import os
from .consts import ProjectPath
from .configs import Configs


__all__ = ['ProjectPath', 'cfg']


cfg = Configs()


def _mkdir_dirs(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


_mkdir_dirs(ProjectPath.LOGS_DIR.value, ProjectPath.CONSOLE_LOGS_DIR.value, ProjectPath.TB_LOGS_DIR.value)
