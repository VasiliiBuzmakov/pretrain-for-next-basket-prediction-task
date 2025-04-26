import torch
import torch.optim as optim
import os
from copy import deepcopy
from utils.utils import import_by_name


def train(model, training_data, validation_data, optimizer, scheduler, opt):
    """ Start training. """
    best_model = deepcopy(model)
    train_epoch = import_by_name(f'models.{opt.model_name}.finetune.{opt.finetune_type}.train_eval_routine', 'train_epoch')
    eval_epoch = import_by_name(f'models.{opt.model_name}.finetune.{opt.finetune_type}.train_eval_routine', 'eval_epoch')
    best_loss = None
    for epoch_i in range(opt.finetune["epoch"]):
        print('[ Epoch', epoch_i, ']')

        train_loss = train_epoch(model, training_data, optimizer, opt)
        eval_loss = eval_epoch(model, validation_data, opt)
        print('  - (finetune: )   train_loss: {train_loss: 8.4f}, '
              ' eval_loss :{eval_loss:8.4f},'
              .format(train_loss=train_loss, eval_loss=eval_loss))

        if best_loss is None or eval_loss < best_loss:
            best_model = deepcopy(model)
            best_loss = eval_loss
        scheduler.step()
    return best_model


def finetune(opt):
    model = opt.model
    prepare_dataloader = import_by_name(f'models.{opt.model_name}.finetune.{opt.finetune_type}.prepare_dataloader', 'prepare_dataloader')
    trainloader, devloader, testloader = prepare_dataloader(opt)

    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.finetune["lr"], betas=opt.finetune["betas"], eps=opt.finetune["eps"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.finetune["scheduler_step"], gamma=opt.finetune["gamma"])

    model = train(model, trainloader, testloader, optimizer, scheduler, opt)

    model.eval()
    model.cpu()
    os.makedirs(opt.model_save_path, exist_ok=True)
    torch.save({"model_weights": model.state_dict()}, opt.model_save_path + f'/{opt.exp_name}_finetune')
    return model
