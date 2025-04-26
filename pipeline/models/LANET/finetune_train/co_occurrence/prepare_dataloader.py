from models.LANET.finetune.co_occurrence.Dataset_co_occurrence import get_dataloader
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
    train_data, num_types = load_data(os.path.join(opt.main_path, opt.data + 'train.pkl'), 'train')
    print("len train data", len(train_data))
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(os.path.join(opt.main_path, opt.data + 'dev.pkl'), 'dev')
    print("len dev data", len(dev_data))
    print('[Info] Loading test data...')
    test_data, _ = load_data(os.path.join(opt.main_path, opt.data + 'test.pkl'), 'test')
    print("len test_data", len(test_data))

    trainloader = get_dataloader(train_data, opt, shuffle=True, train=True)
    devloader = get_dataloader(dev_data, opt, shuffle=False, train=False)
    testloader = get_dataloader(test_data, opt, shuffle=False, train=False)

    return trainloader, devloader, testloader
