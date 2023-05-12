import os
import shutil

import numpy.typing as npt

RESULTS_FOLDER = "results"
PARAMETERS_PATH = "src/parameters.py"


def get_latest_result_folder_number(path=RESULTS_FOLDER) -> int:
    if not os.path.exists(path):
        return 0
    return len(os.listdir(path))


def get_current_result_folder(path=RESULTS_FOLDER) -> str:
    return f"{path}/{get_latest_result_folder_number()}"


def create_result_folder(path=RESULTS_FOLDER) -> None:
    latest_result = get_latest_result_folder_number(path)
    os.makedirs(f"{RESULTS_FOLDER}/{latest_result + 1}")


def copy_parameters(path=RESULTS_FOLDER) -> None:
    latest_result_folder = get_current_result_folder(path)
    destination = f"{latest_result_folder}/parameters.py"
    shutil.copyfile(PARAMETERS_PATH, destination)


def print_solution(chromosome: npt.NDArray) -> str:
    return "".join(chromosome.astype(int).astype(str).tolist())
