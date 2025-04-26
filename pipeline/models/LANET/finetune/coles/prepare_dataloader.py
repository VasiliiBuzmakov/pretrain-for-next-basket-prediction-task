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
    for i in range(len(train_data_)):
        if len(train_data_[i]) >= 3:
            train_data.append(train_data_[i])
    print("len train data", len(train_data))

    print('[Info] Loading dev data...')
    dev_data_, _ = load_data(os.path.join(opt.main_path, opt.data + 'dev.pkl'), 'dev')
    dev_data = []
    for i in range(len(dev_data_)):
        if len(dev_data_[i]) >= 3:
            dev_data.append(dev_data_[i])
    print("len test_data", len(dev_data))

    print('[Info] Loading test data...')
    test_data_, _ = load_data(os.path.join(opt.main_path, opt.data + 'test.pkl'), 'test')
    test_data = []
    for i in range(len(test_data_)):
        if len(test_data_[i]) >= 3:
            test_data.append(test_data_[i])
    train_data.extend(test_data_)
    print("len test_data", len(test_data))
    trainloader = get_dataloader(train_data, opt, shuffle=True, train=True)
    devloader = get_dataloader(dev_data, opt, shuffle=False, train=False)
    testloader = get_dataloader(test_data, opt, shuffle=False, train=False)

    return trainloader, devloader, testloader
