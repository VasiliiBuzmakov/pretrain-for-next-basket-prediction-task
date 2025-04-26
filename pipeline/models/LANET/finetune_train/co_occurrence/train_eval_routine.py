import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from models.LANET.finetune.ContrastiveLoss import ContrastiveLoss
from models.LANET.finetune.BCE import BCELoss
from copy import deepcopy


def multilabel_celoss(x, y):
    mean0 = 1 - x + 10 ** (-9)
    mean1 = x + 10 ** (-9)
    mean1log = torch.log(mean1)
    mean0log = torch.log(mean0)

    logProbTerm1 = y * mean1log
    logProbTerm2 = (1 - y) * mean0log

    logterms = logProbTerm1 + logProbTerm2
    loss = -torch.sum(logterms, dim=1)
    return loss


def train_epoch(model,
                trainloader,
                optimizer,
                opt):
    """ Epoch operation in training phase. """

    model.train()
    avg_loss = []
    lambda_ = opt.lambda_
    finetune_loss_fn = nn.CrossEntropyLoss()
    for batch in tqdm(trainloader, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        time, event_type, co_occurrence = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()
        model.do_finetune = True
        enc_out, _ = model(event_type, time)
        scalars = torch.matmul(enc_out, torch.transpose(enc_out, 1, 2))

        finetune_loss = (finetune_loss_fn(scalars, co_occurrence) + finetune_loss_fn(scalars, torch.transpose(co_occurrence, 1, 2))) / 2

        model.do_finetune = False
        enc_out, non_pad_mask = model(event_type, time)
        a, b, c = enc_out.shape[0], enc_out[:, :-1, :].shape[1], enc_out.shape[2]
        # calculate P*(y_i+1) by mbn:
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a * b, c),
                                          event_type[:, 1:, :].reshape(a * b, c))
        log_loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) / (c * a * torch.sum((torch.sum(non_pad_mask, dim=1) - 1)).item())
        train_loss = log_loss + finetune_loss * lambda_
        avg_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
    return sum(avg_loss) / len(avg_loss)


@torch.inference_mode()
def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    avg_loss = []
    lambda_ = opt.lambda_
    finetune_loss_fn = nn.CrossEntropyLoss()
    for batch in tqdm(validation_data, mininterval=2,
                      desc='  - (Evaling)   ', leave=False):
        time, event_type, co_occurrence = map(lambda x: x.to(opt.device), batch)

        model.do_finetune = True
        enc_out, _ = model(event_type, time)
        scalars = torch.matmul(enc_out, torch.transpose(enc_out, 1, 2))

        finetune_loss = (finetune_loss_fn(scalars, co_occurrence) + finetune_loss_fn(scalars, torch.transpose(co_occurrence, 1, 2))) / 2

        model.do_finetune = False
        enc_out, non_pad_mask = model(event_type, time)
        a, b, c = enc_out.shape[0], enc_out[:, :-1, :].shape[1], enc_out.shape[2]
        # calculate P*(y_i+1) by mbn:
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a * b, c),
                                          event_type[:, 1:, :].reshape(a * b, c))
        log_loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) / (c * a * torch.sum((torch.sum(non_pad_mask, dim=1) - 1)).item())
        eval_loss = log_loss + finetune_loss * lambda_
        avg_loss.append(eval_loss.item())
    return sum(avg_loss) / len(avg_loss)
