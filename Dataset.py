import torch
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
    # print(t_train.shape, t_test.shape)
    return x_train, t_train, x_test, t_test

class Dataset():
    def __init__(self, device):
        self.x_train = torch.tensor(get_data()[0], device=device)
        self.device = device
    
    def __len__(self):
        return self.x_train.shape[0]
    
    def __getitem__(self, index):
        # if index < self.x_train.shape[0]:
        #     return self.x_train[index], torch.tensor(1.0, device=self.device)
        # else:
        #     return self.G.getImage("test"), torch.tensor(0.0, device=self.device)
        return self.x_train[index], torch.tensor(1.0, device=self.device)