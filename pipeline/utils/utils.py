import torch
import random
import numpy as np
import importlib
import os

def set_random_seed(seed: int = 0) -> None:
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def import_by_name(module_name, object_name):
    module = importlib.import_module(module_name)
    return getattr(module, object_name)

def save_to_csv(y, type, opt):
    'type is pred_test or pred_valid'
    if opt.ablation:
        path_to_save = f'model_pred_and_gt/{opt.model_name}_ablation_{opt.ablation_index}/{opt.dataset_name}/run_{opt.seed}/{type}'
    else:
        path_to_save = f'model_pred_and_gt/{opt.model_name}/{opt.dataset_name}/run_{opt.seed}/{type}'

    os.makedirs(path_to_save, exist_ok=True)
    np.savetxt(os.path.join(path_to_save, 'data.csv'), y.numpy(), delimiter=',')


def append_to_txt(text: str):
    file_path = 'utils/history.txt'

    text = '\n' + text + '\n'
    with open(file_path, 'a') as file:
        file.write(text)