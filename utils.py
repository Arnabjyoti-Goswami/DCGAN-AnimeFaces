import os
import shutil
import subprocess

import torch


def get_default_device():
    """
    Return a device (cuda or cpu) based on the availability of GPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """
    Move tensor(s) to chosen device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """
    A wrapper around the torch dataloader, that moves data to the required device before returning the data for an iteration in the __iter__ method
    """

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def get_environment_type() -> str:
    """
    Returns 'colab', 'kaggle' or 'local' based on the environment
    """
    colab_env_var = "COLAB_RELEASE_TAG"
    kaggle_env_var = "KAGGLE_DOCKER_IMAGE"

    if os.getenv(colab_env_var):
        return "colab"

    if os.getenv(kaggle_env_var):
        return "kaggle"

    return "local"


def create_directory(path: str) -> None:
    """
    Create a folder if it doesn't already exist
    """
    if os.path.exists(path):
        print(f"Directory {path} already exists")
        return
    try:
        os.makedirs(path)
        print(f"Directory {path} created successfully")
    except Exception as e:
        print(f"An error occurred while creating the directory {path}")
        print(e)


def move_path(src_path: str, dst_path: str):
    """
    Move a file or directory from a source path to a target path
    """
    try:
        shutil.move(src_path, dst_path)
    except Exception as e:
        print(f"An error occurred while moving {src_path} to {dst_path}")
        print(e)


def run_command(command) -> str:
    """
    Run a shell command
    """
    try:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        output = stdout.decode("utf-8") + stderr.decode("utf-8")
        return output
    except Exception as e:
        print(f"An error occurred while running the command {command}")
        print(e)
        return ""


class CellStopExecution(Exception):
    def _render_traceback_(self):
        return []
