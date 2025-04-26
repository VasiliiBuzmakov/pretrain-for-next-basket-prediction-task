import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


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


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    avg_loss = []
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        event_time, event_type, time_gap = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()
        enc_out, non_pad_mask = model(event_type, event_time, False, False, False)
        a, b, c = enc_out.shape[0], enc_out[:, :-1, :].shape[1], enc_out.shape[2]
        # calculate P*(y_i+1) by mbn:
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a * b, c),
                                          event_type[:, 1:, :].reshape(a * b, c))
        loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) / (a * torch.sum((torch.sum(non_pad_mask, dim=1) - 1)).item())
        avg_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return sum(avg_loss) / len(avg_loss)


@torch.inference_mode()
def eval_epoch(model, validation_data, opt, val_flag=False, alpha=0.5):
    """ Epoch operation in evaluation phase. """

    model.eval()
    pred_label = []
    true_label = []
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc=f'  - ({type}) ', leave=False):
            """ prepare data """

            # _, _, original_time, original_event_type = map(lambda x: x.to(opt.device), batch)
            original_time, original_event_type, _ = map(lambda x: x.to(opt.device), batch)
            enc_out, non_pad_mask = model(original_event_type, original_time, False, False, False)

            last_pred_idx = (non_pad_mask.sum(dim=1) - 1).long()
            batch_indices = torch.arange(enc_out.size(0)).to(enc_out.device)
            pred_type = enc_out[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last prediction for each sequence
            # pred_label += list(pred_type.detach().cpu())
            pred_label.append(pred_type.detach().cpu())

            # Extract the true value for the corresponding last predictions
            true_type = original_event_type[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last true value for each sequence
            # true_label += list(true_type.detach().cpu())
            true_label.append(true_type.detach().cpu())

        true_label = torch.cat(true_label, dim=0)
        pred_label = torch.cat(pred_label, dim=0)

    tasks_with_non_trivial_targets = np.where(true_label.sum(axis=0) != 0)[0]
    y_pred_copy = pred_label[:, tasks_with_non_trivial_targets].numpy()
    y_true_copy = true_label[:, tasks_with_non_trivial_targets].numpy()
    roc_auc = roc_auc_score(y_true=y_true_copy, y_score=y_pred_copy, average='weighted')
    best_f1_metric = f1_score(y_true=y_true_copy, y_pred=(y_pred_copy > alpha), average='weighted')
    best_alpha = 0.5
    if val_flag:
        for alpha in np.linspace(0.1, 1, num=15):
            f1_metric = f1_score(y_true=y_true_copy, y_pred=(y_pred_copy > alpha), average='weighted')
            if f1_metric > best_f1_metric:
                best_f1_metric = f1_metric
                best_alpha = alpha
    return 0, roc_auc, best_f1_metric, best_alpha


def evaluate(model, dataloader, opt, type) -> None:
    model.eval()
    pred_label = []
    true_label = []
    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=2,
                          desc=f'  - ({type}) ', leave=False):
            """ prepare data """

            event_time, event_type, time_gap = map(lambda x: x.to(opt.device), batch)

            enc_out, non_pad_mask = model(event_type, event_time)

            last_pred_idx = (non_pad_mask.sum(dim=1) - 1).long()
            batch_indices = torch.arange(enc_out.size(0)).to(enc_out.device)
            pred_type = enc_out[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last prediction for each sequence
            pred_label += list(pred_type.detach().cpu())

            # Extract the true value for the corresponding last predictions
            true_type = event_type[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last true value for each sequence
            true_label += list(true_type.detach().cpu())

            if opt.debug_stop:
                break

    y_pred = torch.cat(pred_label, dim=0).reshape(-1, model.num_types)
    y_true = torch.cat(true_label, dim=0).reshape(-1, model.num_types)
