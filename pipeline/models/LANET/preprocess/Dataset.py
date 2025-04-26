import numpy as np
import torch
import torch.utils.data

from ..transformer import Constants
from typing import Any, List, Optional

class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[elem['type_event'] for elem in inst] for inst in data]
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx]

class MyCollator:
    def __init__(self, num_types):
        self.num_types = num_types

    def __call__(self, batch):
        """ Collate function, as required by PyTorch. """

        time, time_gap, event_type = list(zip(*batch))
        time = self.pad_time(time)
        time_gap = self.pad_time(time_gap)
        event_type = self.pad_type(event_type)

        return time, event_type, time_gap

    def pad_time(self, batch):
        """ Pad the instance to the max seq length in batch. """

        max_len = max(len(inst) for inst in batch)

        batch_seq = np.array([
            inst + [self.num_types] * (max_len - len(inst))
            for inst in batch])

        return torch.tensor(batch_seq, dtype=torch.float32)

    def pad_type(self, insts):
        """ Pad the instance to the max seq length in batch. """

        max_len = max(len(inst) for inst in insts)

        batch_seq = np.array([
            inst + [[self.num_types] * self.num_types] * (max_len - len(inst))
            for inst in insts])

        return torch.tensor(batch_seq, dtype=torch.long)


def get_dataloader(data, opt, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    my_collator = MyCollator(opt.num_types)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.train["batch_size"],
        collate_fn=my_collator,
        shuffle=shuffle,
        pin_memory=True
    )
    return dl