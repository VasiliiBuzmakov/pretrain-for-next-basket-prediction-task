import numpy as np
import torch
import torch.utils.data

PAD_INDEX = -1


def get_non_pad_mask(seq, pad_idx):
    assert seq.dim() == 2
    return seq.ne(pad_idx).type(torch.float).unsqueeze(-1)


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
        events = []
        for batch_ind in range(len(event_type)):
            batch_indices = []
            for seq_ind in range(len(event_type[batch_ind])):
                unit_indices = np.where(event_type[batch_ind][seq_ind] == 1)[0]
                batch_indices.append(unit_indices.tolist())
            events.append(batch_indices)
        max_len = max(len(event) for event in events)
        return events, self.pad_type(event_type), get_non_pad_mask(self.pad_time(time), PAD_INDEX), max_len

    def pad_time(self, batch):
        """ Pad the instance to the max seq length in batch. """

        max_len = max(len(inst) for inst in batch)

        batch_seq = np.array([
            inst + [PAD_INDEX] * (max_len - len(inst))
            for inst in batch])

        return torch.tensor(batch_seq, dtype=torch.float32)

    def pad_type(self, insts):
        """ Pad the instance to the max seq length in batch. """

        max_len = max(len(inst) for inst in insts)

        batch_seq = np.array([
            inst + [[self.num_types] * self.num_types] * (max_len - len(inst))
            for inst in insts])

        return torch.tensor(batch_seq, dtype=torch.float32)


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