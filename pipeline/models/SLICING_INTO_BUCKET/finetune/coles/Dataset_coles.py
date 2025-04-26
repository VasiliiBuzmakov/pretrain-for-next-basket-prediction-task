import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence


def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        self.event_type = [[elem['type_event'] for elem in inst] for inst in data]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.time[idx], self.time_gap[idx], self.event_type[idx]


class MyCollator:
    def __init__(self, num_types, sample_size, train=True):
        self.num_types = num_types
        self.sample_size = sample_size
        self.train = train

    def generate_indexes(self, high, low=0):
        i, j = 0, 0
        while abs(j - i) <= high // 3:
            i = torch.randint(low=0, high=high, size=(1,)).item()
            j = torch.randint(low=0, high=high, size=(1,)).item()
            # if not self.train:
            #     break
        i, j = min(i, j), max(i, j)
        return i, j

    def generate_samples(self, events, events_generated, index):
        for _ in range(self.sample_size):
            bucket = np.random.choice(len(events[index]))
            i, j = self.generate_indexes(len(events[index][bucket])+1)
            events_generated.append(events[index][bucket][i:j])

            positive_bucket = np.random.choice(len(events[index]))
            i, j = self.generate_indexes(len(events[index][positive_bucket])+1)
            events_generated.append(events[index][positive_bucket][i:j])

            valid_numbers = np.setdiff1d(np.arange(len(events)), [index])
            niggative = np.random.choice(valid_numbers)
            negative_bucket = np.random.choice(len(events[niggative]))
            i, j = self.generate_indexes(len(events[niggative][negative_bucket])+1)
            events_generated.append(events[niggative][negative_bucket][i:j])

    def __call__(self, batch):
        """ Collate function, as required by PyTorch. """
        time, time_gap, event_type = list(zip(*batch))
        events = []
        for batch_ind in range(len(event_type)):
            batch_indices = []
            for seq_ind in range(len(event_type[batch_ind])):
                unit_indices = np.where(event_type[batch_ind][seq_ind] == 1)[0]
                batch_indices.append(unit_indices.tolist())
            events.append(batch_indices)
        events_generated = []
        for i in range(len(time)):
            self.generate_samples(events, events_generated, i)
        max_len = max(len(event) for event in events_generated)
        padded_tensor = pad_sequence(
            [torch.tensor(arr, dtype=torch.int32) for arr in events_generated],
            batch_first=True,
            padding_value=self.num_types
        )
        return padded_tensor, get_non_pad_mask(padded_tensor, self.num_types), max_len


def get_dataloader(data, opt, shuffle=True, train=True):
    ds = EventData(data)
    my_collator = MyCollator(opt.num_types, opt.finetune["sample_size"], train)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.finetune["batch_size"],
        collate_fn=my_collator,
        shuffle=shuffle,
        pin_memory=True
    )
    return dl
