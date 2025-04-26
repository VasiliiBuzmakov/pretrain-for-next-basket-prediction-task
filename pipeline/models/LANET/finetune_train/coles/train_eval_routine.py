import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from models.LANET.finetune.ContrastiveLoss import ContrastiveLoss
from models.LANET.finetune.BCE import BCELoss


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
    finetune_loss_fn = nn.TripletMarginLoss()
    for batch in tqdm(trainloader, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        time_generated, event_type_generated, original_time, original_event_type = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        model.do_finetune = True
        enc_out, non_pad_mask = model(event_type_generated, time_generated)
        b, n = enc_out.size(0), enc_out.size(1)
        anchors = torch.zeros(b // 3, n)
        positives = torch.zeros(b // 3, n)
        negatives = torch.zeros(b // 3, n)
        i = 0
        ind = 0
        while i < b:
            anchors[ind] = enc_out[i]
            positives[ind] = enc_out[i + 1]
            negatives[ind] = enc_out[i + 2]
            ind += 1
            i += 3
        finetune_loss = finetune_loss_fn(anchors, positives, negatives)

        model.do_finetune = False
        enc_out, non_pad_mask = model(original_event_type, original_time)
        a, b, c = enc_out.shape[0], enc_out[:, :-1, :].shape[1], enc_out.shape[2]

        # calculate P*(y_i+1) by mbn:
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a * b, c),
                                          original_event_type[:, 1:, :].reshape(a * b, c))

        train_loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) / (c * a * torch.sum((torch.sum(non_pad_mask, dim=1) - 1)).item())
        res = train_loss + finetune_loss * lambda_
        avg_loss.append(res.item())
        res.backward()
        optimizer.step()
    return sum(avg_loss) / len(avg_loss)


@torch.inference_mode()
def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    lambda_ = opt.lambda_
    avg_loss = []
    finetune_loss_fn = nn.TripletMarginLoss()
    for batch in tqdm(validation_data, mininterval=2,
                      desc='  - (Evaling)   ', leave=False):
        time_generated, event_type_generated, original_time, original_event_type = map(lambda x: x.to(opt.device), batch)

        model.do_finetune = True
        enc_out, non_pad_mask = model(event_type_generated, time_generated)
        b, n = enc_out.size(0), enc_out.size(1)
        anchors = torch.zeros(b // 3, n)
        positives = torch.zeros(b // 3, n)
        negatives = torch.zeros(b // 3, n)
        i = 0
        ind = 0
        while i < b:
            anchors[ind] = enc_out[i]
            positives[ind] = enc_out[i + 1]
            negatives[ind] = enc_out[i + 2]
            ind += 1
            i += 3
        finetune_loss = finetune_loss_fn(anchors, positives, negatives)

        model.do_finetune = False
        enc_out, non_pad_mask = model(original_event_type, original_time)
        a, b, c = enc_out.shape[0], enc_out[:, :-1, :].shape[1], enc_out.shape[2]

        # calculate P*(y_i+1) by mbn:
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a * b, c), original_event_type[:, 1:, :].reshape(a * b, c))
        train_loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) / (c * enc_out.shape[1] * torch.sum((torch.sum(non_pad_mask, dim=1) - 1)).item())
        res = train_loss + finetune_loss * lambda_
        avg_loss.append(res.item())

    return sum(avg_loss) / len(avg_loss)
