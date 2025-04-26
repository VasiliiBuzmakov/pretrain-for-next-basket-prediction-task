import torch
import torch.optim as optim
import os
from copy import deepcopy
from utils.utils import import_by_name

def train_(model,
           trainloader,
           testloader,
           optimizer,
           scheduler,
           opt):
    """ Start training. """
    best_model = deepcopy(model.state_dict())
    best_eval_loss = 10 ** 10
    best_roc_auc = 0
    train_epoch = import_by_name(f'models.{opt.model_name}.finetune_train.{opt.finetune_type}.train_eval_routine', 'train_epoch')
    eval_epoch = import_by_name(f'models.{opt.model_name}.finetune_train.{opt.finetune_type}.train_eval_routine', 'eval_epoch')
    roc_auc_val = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'eval_epoch')

    for epoch_i in range(opt.finetune["epoch"]):
        print('[ Epoch', epoch_i, ']')

        train_loss = train_epoch(model,
                                 trainloader,
                                 optimizer,
                                 opt)

        scheduler.step()
        eval_loss = eval_epoch(model, testloader, opt)
        print('  - (train: )   train_loss: {train_loss: 8.6f}, '
              ' eval_loss :{eval_loss:8.6f},'
              .format(train_loss=train_loss, eval_loss=eval_loss))

        model.do_finetune = False
        valid_event, valid_roc_auc, f1_metric = roc_auc_val(model, testloader, opt)
        print('  - (test)    nll: {ll: 8.6f}, '
              ' roc auc : {type:8.6f},'
              ' f1 metric : {f1:8.6f},'
              .format(ll=valid_event, type=valid_roc_auc, f1=f1_metric))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_model = deepcopy(model.state_dict())
        if best_roc_auc < valid_roc_auc:
            best_roc_auc = valid_roc_auc
        print(f"best roc auc val: {best_roc_auc}")
    return best_model


def train_with_finetune(opt):
    model = opt.model
    prepare_dataloader = import_by_name(f'models.{opt.model_name}.finetune_train.{opt.finetune_type}.prepare_dataloader', 'prepare_dataloader')
    trainloader, devloader, testloader = prepare_dataloader(opt)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.train["lr"], betas=opt.train["betas"], eps=opt.train["eps"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.train["scheduler_step"], gamma=opt.train["gamma"])

    model = train_(model,
                  trainloader,
                  testloader,
                  optimizer,
                  scheduler,
                  opt)

    model.eval()
    model.cpu()
    os.makedirs(opt.model_save_path, exist_ok=True)
    torch.save({"model_weights": model.state_dict()}, opt.model_save_path + f'/{opt.exp_name}_finetune')
    return model
