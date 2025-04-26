import torch
import torch.optim as optim
import os
from finetune import finetune
from train import train
from finetune_train import train_with_finetune
from copy import deepcopy
from utils.utils import import_by_name, append_to_txt, set_random_seed
from utils.load_config import config
import time


def get_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def load_checkpoint(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_weights'])
    return model


def main(conf):
    """ Main function. 
    Parse config file, create model, create dataloader for model, train and save model
    """
    opt = deepcopy(conf)

    opt.device = torch.device(f'{opt.device_name}:{opt.device_id}')
    opt.model_save_path = f"saved_models/{opt.model_name}_train_/{opt.dataset_name}"
    opt.main_path = os.path.dirname(os.path.realpath(__file__))

    set_random_seed(opt.seed)
    create_model = import_by_name(f'models.{opt.model_name}.model_creation', 'create_model')
    model = create_model(opt)

    if opt.upload_model_config["upload_model"]:
        model = load_checkpoint(model, os.path.join(opt.model_save_path, opt.upload_model_config["exp_name"]))

    model.to(opt.device)
    opt.model = model

    if opt.do_finetune:
        opt.model = finetune(opt)
    if opt.do_train_with_finetune:
        opt.model = train_with_finetune(opt)
        return
    if opt.do_train:
        opt.model.do_finetune = False
        opt.model = train(opt)


if __name__ == '__main__':
    model_names = deepcopy(get_list(config.model_name))
    dataset_names = deepcopy(get_list(config.dataset_name))
    seeds = deepcopy(get_list(config.seed))

    start = time.time()
    for dataset in dataset_names:
        for model in model_names:
            for seed in seeds:
                config.modify_config(model_name=model, dataset_name=dataset, seed=seed)
                main(conf=config)

    end = time.time()
    print("total training time is {}".format(end - start))
