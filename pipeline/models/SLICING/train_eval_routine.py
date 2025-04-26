import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.nn as nn


def generate_future_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class CustomMultiLabelLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomMultiLabelLoss, self).__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Без агрегации

    def forward(self, outputs, targets, mask):
        loss = self.bce_loss(outputs, targets) * mask
        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    avg_loss = []
    loss_fn = CustomMultiLabelLoss("mean")
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        events, targets, non_pad_mask, max_len = batch
        not_to_lookup_future_mask = generate_future_mask(max_len)
        optimizer.zero_grad()
        enc_out = model(events, max_len, not_to_lookup_future_mask, (1 - non_pad_mask).squeeze(-1))
        loss = loss_fn(enc_out[:, :-1, :], targets[:, 1:, :], mask=non_pad_mask[:, 1:, :]) / len(events)
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
    for batch in tqdm(validation_data, mininterval=2,
                      desc=f'  - ({type}) ', leave=False):
        """ prepare data """

        events, targets, non_pad_mask, max_len = batch
        not_to_lookup_future_mask = generate_future_mask(max_len)
        enc_out = model(events, max_len, not_to_lookup_future_mask, (1 - non_pad_mask).squeeze(-1))
        last_pred_idx = (non_pad_mask.sum(dim=1) - 1).long()
        batch_indices = torch.arange(enc_out.size(0)).to(enc_out.device)
        pred_type = enc_out[batch_indices, last_pred_idx.squeeze(), :]

        # Store the last prediction for each sequence
        pred_label.append(F.sigmoid(pred_type).detach().cpu())

        # Extract the true value for the corresponding last predictions
        true_type = targets[batch_indices, last_pred_idx.squeeze(), :]

        # Store the last true value for each sequence
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
            pred_label += list(F.sigmoid(pred_type.detach().cpu()))

            # Extract the true value for the corresponding last predictions
            true_type = event_type[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last true value for each sequence
            true_label += list(true_type.detach().cpu())

            if opt.debug_stop:
                break

    # y_pred = torch.cat(pred_label, dim=0).reshape(-1, model.num_types)
    # y_true = torch.cat(true_label, dim=0).reshape(-1, model.num_types)
    #
    # save_to_csv(y_pred, f'pred_{type}', opt)
    # save_to_csv(y_true, f'gt_{type}', opt)
