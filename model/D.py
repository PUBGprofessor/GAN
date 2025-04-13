import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.Conv1 = nn.Conv2d(1, 6, 5)
        self.Conv2 = nn.Conv2d(6, 16, 5)
        
        # Dropout 层
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(16 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        batch = x.size()[0]
        x = F.relu(self.Conv1(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout1(x)

        x = F.relu(self.Conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = self.dropout2(x)

        x = x.view(batch, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x

    def predict(self, x, s="test"):
        if s not in ["train", "test"]:
            raise ValueError("getImage:trian or test")
        
        # 记录原状态
        if self.training:
            preState = True
        else:
            preState = False
        
        if s == "train":
            self.train()
        else:
            self.eval()
        y = self.forward(x)

        # 回归原状态
        if preState == True:
            self.train()
        else:
            self.eval()
        return torch.sigmoid(y)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = torch.argmax(y, axis=1)
        if t.ndim != 1 : t = torch.argmax(t, axis=1)
        accuracy = torch.sum(y == t) / float(x.shape[0])
        return accuracy