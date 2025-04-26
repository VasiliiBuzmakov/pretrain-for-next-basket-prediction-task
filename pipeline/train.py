import torch
import torch.optim as optim
import os
from copy import deepcopy

from utils.utils import import_by_name
import time


def get_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_(model, training_data, validation_data, test_data, opt):
    """ Start training. """
    best_auc_roc = 0
    impatient = 0
    best_model = deepcopy(model.state_dict())
    last_roc_auc = 0
    train_epoch = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'train_epoch')
    eval_epoch = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'eval_epoch')

    if opt.do_finetune and opt.epoch_count_freeze_weights != 0:
        layer_name_not_to_freeze = "linear"
        for name, param in model.named_parameters():
            param.requires_grad = False
            if layer_name_not_to_freeze in name:
                param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.train["lr"], betas=opt.train["betas"], eps=opt.train["eps"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.train["scheduler_step"], gamma=opt.train["gamma"])

    for epoch_i in range(opt.train["epoch"]):
        if opt.do_finetune and opt.epoch_count_freeze_weights == epoch_i:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=get_lr(optimizer), betas=opt.train["betas"], eps=opt.train["eps"])
            scheduler = optim.lr_scheduler.StepLR(optimizer, opt.train["scheduler_step"], gamma=opt.train["gamma"])

        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_event = train_epoch(model, training_data, optimizer, opt)
        print('  - (Train)    negative loglikelihood: {ll: 8.6f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_roc_auc, f1_metric, alpha = eval_epoch(model, validation_data, opt, val_flag=True)
        print('  - (dev)    nll: {ll: 8.6f}, '
              ' roc auc : {type:8.6f},'
              ' f1 metric : {f1:8.6f},'
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_roc_auc, f1=f1_metric, elapse=(time.time() - start) / 60))

        start = time.time()
        test_event, test_roc_auc, f1_metric, alpha = eval_epoch(model, test_data, opt, alpha=alpha)
        print('  - (test)    nll: {ll: 8.6f}, '
              ' roc auc :{type:8.6f},'
              ' f1 metric :{f1:8.6f},'
              'elapse: {elapse:3.3f} min'
              .format(ll=test_event, type=test_roc_auc, f1=f1_metric, elapse=(time.time() - start) / 60))

        if ((best_auc_roc - valid_roc_auc) < opt.train["early_stop_thr"]) or abs(
                last_roc_auc - valid_roc_auc) < opt.train["early_stop_thr"]:
            impatient += 1
        else:
            impatient = 0

        if best_auc_roc < valid_roc_auc:
            best_auc_roc = valid_roc_auc
            best_model = deepcopy(model)
            impatient = 0

        if impatient >= 20:
            print(f'Breaking due to early stopping at epoch {epoch_i}')
            break

        scheduler.step()
        print(f'Impatience: {impatient}')
        print(f'Best roc auc: {best_auc_roc}')
        last_roc_auc = valid_roc_auc

    return best_model, best_auc_roc


def train(opt):
    model = opt.model
    prepare_dataloader = import_by_name(f'models.{opt.model_name}.prepare_dataloader', 'prepare_dataloader')
    trainloader, devloader, testloader = prepare_dataloader(opt)
    model, best_roc_auc = train_(model, trainloader, devloader, testloader, opt)
    model.eval()
    model.cpu()
    os.makedirs(opt.model_save_path, exist_ok=True)
    torch.save({"model_weights": model.state_dict()}, opt.model_save_path + f'/{opt.exp_name}_train')
    return model
