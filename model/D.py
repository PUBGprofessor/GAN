import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        # Convention kernel 1：1->6 5x5
        self.Conv1 = nn.Conv2d(1, 6, 5)
        # Convention kernel 2：6->16 5x5
        self.Conv2 = nn.Conv2d(6, 4, 5)

        # self.fc1 = nn.Linear(16*4*4, 64)
        self.fc1 = nn.Linear(4*4*4, 1)
        # self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 输入x:(batch_size, 1, 28, 28)
        batch =x.size()[0]
        x = F.max_pool2d(F.relu(self.Conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.Conv2(x)), (2,2))
        x = x.view(batch, -1)
        # x = F.relu(self.fc1(x))

        return self.fc1(x)
        # return self.fc2(x)

    def predict(self, x):
        y = self.forward(x)
        return F.sigmoid(y)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = torch.argmax(y, axis=1)
        if t.ndim != 1 : t = torch.argmax(t, axis=1)
        accuracy = torch.sum(y == t) / float(x.shape[0])
        return accuracy