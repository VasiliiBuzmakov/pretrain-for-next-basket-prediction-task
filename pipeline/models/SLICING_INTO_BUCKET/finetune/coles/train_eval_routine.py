import torch
from tqdm import tqdm
import torch.nn as nn


def generate_future_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    avg_loss = []
    loss_fn = nn.TripletMarginLoss()
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        events, non_pad_mask, max_len = batch
        not_to_lookup_future_mask = generate_future_mask(max_len)
        optimizer.zero_grad()
        enc_out = model(events, max_len, not_to_lookup_future_mask, (1 - non_pad_mask).squeeze(-1))
        b, n = enc_out.size(0), enc_out.size(1)
        # anchors = torch.zeros(b // 3 * 2, n)
        # items_to_anchor = torch.zeros(b // 3 * 2, n)
        # labels = torch.zeros(b // 3 * 2, 1)
        anchors = torch.zeros(b // 3, n)
        positives = torch.zeros(b // 3, n)
        negatives = torch.zeros(b // 3, n)

        i = 0
        ind = 0
        while i < b:
            # anchors[ind] = enc_out[i]
            # items_to_anchor[ind] = enc_out[i + 1]
            # labels[ind] = 1
            # anchors[ind + 1] = enc_out[i]
            # items_to_anchor[ind + 1] = enc_out[i + 2]
            # labels[ind + 1] = 0
            # ind += 2
            # i += 3
            anchors[ind] = enc_out[i]
            positives[ind] = enc_out[i + 1]
            negatives[ind] = enc_out[i + 2]
            ind += 1
            i += 3
        train_loss = loss_fn(anchors, positives, negatives)

        # train_loss = loss_fn(anchors, items_to_anchor, labels)
        avg_loss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
    return sum(avg_loss) / len(avg_loss)


@torch.inference_mode()
def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    avg_loss = []
    loss_fn = nn.TripletMarginLoss()
    for batch in tqdm(validation_data, mininterval=2, desc='  - (Evaling)   ', leave=False):
        events, non_pad_mask, max_len = batch
        not_to_lookup_future_mask = generate_future_mask(max_len)
        enc_out = model(events, max_len, not_to_lookup_future_mask, (1 - non_pad_mask).squeeze(-1))
        b, n = enc_out.size(0), enc_out.size(1)
        # anchors = torch.zeros(b // 3 * 2, n)
        # items_to_anchor = torch.zeros(b // 3 * 2, n)
        # labels = torch.zeros(b // 3 * 2, 1)
        anchors = torch.zeros(b // 3, n)
        positives = torch.zeros(b // 3, n)
        negatives = torch.zeros(b // 3, n)

        i = 0
        ind = 0
        while i < b:
            # anchors[ind] = enc_out[i]
            # items_to_anchor[ind] = enc_out[i + 1]
            # labels[ind] = 1
            # anchors[ind + 1] = enc_out[i]
            # items_to_anchor[ind + 1] = enc_out[i + 2]
            # labels[ind + 1] = 0
            # ind += 2
            # i += 3
            anchors[ind] = enc_out[i]
            positives[ind] = enc_out[i + 1]
            negatives[ind] = enc_out[i + 2]
            ind += 1
            i += 3
        eval_loss = loss_fn(anchors, positives, negatives)
        # eval_loss = loss_fn(anchors, items_to_anchor, labels)
        avg_loss.append(eval_loss.item())
    return sum(avg_loss) / len(avg_loss)
