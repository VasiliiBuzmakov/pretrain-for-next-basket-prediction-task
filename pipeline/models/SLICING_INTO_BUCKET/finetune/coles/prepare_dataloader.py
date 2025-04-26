from .Dataset_coles import get_dataloader
import pickle
import os


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading train data...')
    train_data_, num_types = load_data(os.path.join(opt.main_path, opt.data + 'train.pkl'), 'train')
    train_data = []
    for user in train_data_:
        lst = list()
        for seq in user:
            if sum(seq["type_event"]) < 4:
                lst.append(seq)
        if len(lst) > 3:
            train_data.append(lst)
    print("len train data", len(train_data))

    print('[Info] Loading dev data...')
    dev_data_, _ = load_data(os.path.join(opt.main_path, opt.data + 'dev.pkl'), 'dev')
    dev_data = []
    for user in dev_data_:
        lst = list()
        for seq in user:
            if sum(seq["type_event"]) < 4:
                lst.append(seq)
        if len(lst) > 3:
            dev_data.append(lst)
    print("len test_data", len(dev_data))

    print('[Info] Loading test data...')
    test_data_, _ = load_data(os.path.join(opt.main_path, opt.data + 'test.pkl'), 'test')
    test_data = []
    for user in test_data_:
        lst = list()
        for seq in user:
            if sum(seq["type_event"]) < 4:
                lst.append(seq)
        if len(lst) != 0:
            test_data.append(lst)
    print("len test_data", len(test_data))

    trainloader = get_dataloader(train_data, opt, shuffle=True, train=True)
    devloader = get_dataloader(dev_data, opt, shuffle=False, train=False)
    testloader = get_dataloader(test_data, opt, shuffle=False, train=False)
    return trainloader, devloader, testloader
