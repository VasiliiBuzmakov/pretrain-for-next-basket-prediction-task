from models.LANET.finetune.coles.Dataset_coles  import get_dataloader
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
    # train_data = []
    # for i in range(len(train_data_)):
    #     if len(train_data_[i]) >= 10:
    #         train_data.append(train_data_[i])
    print('[Info] Loading dev data...')
    dev_data, _ = load_data(os.path.join(opt.main_path, opt.data + 'dev.pkl'), 'dev')
    print('[Info] Loading test data...')
    test_data, _ = load_data(os.path.join(opt.main_path, opt.data + 'test.pkl'), 'test')

    trainloader = get_dataloader(train_data, opt, shuffle=True, train=True)
    devloader = get_dataloader(dev_data, opt, shuffle=False, train=False)
    testloader = get_dataloader(test_data, opt, shuffle=False, train=False)
    return trainloader, devloader, testloader