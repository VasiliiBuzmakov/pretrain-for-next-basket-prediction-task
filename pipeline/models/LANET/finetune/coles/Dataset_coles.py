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
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        self.event_type = [[elem['type_event'] for elem in inst] for inst in data]
        self.length = len(data)
        self.user_element_frequency = []
        for user in self.event_type:
            occurrence = [0 for _ in range(len(user[0]))]
            for binary_mask in user:
                for i in range(len(binary_mask)):
                    occurrence[i] += binary_mask[i]
            self.user_element_frequency.append(occurrence)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.user_element_frequency[idx]


class MyCollator:
    def __init__(self, num_types, sample_size, train=True):
        self.num_types = num_types
        self.sample_size = sample_size
        self.train = train

    def generate_indexes(self, high, low=0):
        i, j = 0, 0
        while i == j or abs(j - i) <= high // 3:
            i = torch.randint(low=0, high=high, size=(1,)).item()
            j = torch.randint(low=0, high=high, size=(1,)).item()
            # if not self.train:
            #     break
        i, j = min(i, j), max(i, j)
        return i, j

    def generate_samples(self, time, events, time_generated, events_generated, index):
        for _ in range(self.sample_size):
            i, j = self.generate_indexes(len(time[index]))
            time_generated.append(time[index][i:j])
            events_generated.append(events[index][i:j])

            i, j = self.generate_indexes(len(time[index]))
            time_generated.append(time[index][i:j])
            events_generated.append(events[index][i:j])

            valid_numbers = np.setdiff1d(np.arange(len(time)), [index])
            niggative = np.random.choice(valid_numbers)
            i, j = self.generate_indexes(len(time[niggative]))
            time_generated.append(time[niggative][i:j])
            events_generated.append(events[niggative][i:j])

    def __call__(self, batch):
        """ Collate function, as required by PyTorch. """
        time, time_gap, event_type, user_element_frequency = list(zip(*batch))
        events = []
        for batch_ind in range(len(event_type)):
            batch_indices = []
            for seq_ind in range(len(event_type[batch_ind])):
                unit_indices = np.where(event_type[batch_ind][seq_ind] == 1)[0]
                batch_indices.append(unit_indices.tolist())
            events.append(batch_indices)
        time_generated = []
        events_generated = []
        for i in range(len(time)):
            self.generate_samples(time, events, time_generated, events_generated, i)
        max_len = max(len(event) for event in events_generated)
        return events_generated, time_generated, get_non_pad_mask(self.pad_time(time_generated), self.num_types), max_len

    def pad_time(self, batch):
        """ Pad the instance to the max seq length in batch. """
        max_len = max(len(inst) for inst in batch)
        batch_seq = np.array([
            inst + [self.num_types] * (max_len - len(inst))
            for inst in batch])
        return torch.tensor(batch_seq, dtype=torch.float32)


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
