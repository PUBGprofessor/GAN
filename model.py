import torch
import torch.nn as nn
import torch.nn.functional as F

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        # Convention kernel 1：1->6 5x5
        self.Conv1 = nn.Conv2d(1, 6, 5)
        # Convention kernel 2：6->16 5x5
        self.Conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*4*4, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch =x.size()[0]
        x = F.max_pool2d(F.relu(self.Conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.Conv2(x)), (2,2))
        x = x.view(batch, -1)
        x = F.relu(self.fc1(x))

        return self.fc2(x)

    def predict(self, x):
        y = self.forward(x)
        return F.sigmoid(y)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = torch.argmax(y, axis=1)
        if t.ndim != 1 : t = torch.argmax(t, axis=1)
        accuracy = torch.sum(y == t) / float(x.shape[0])
        return accuracy
    
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(8)

        self.Conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.BN2 = nn.BatchNorm2d(32)

        self.Conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.BN3 = nn.BatchNorm2d(16)

        self.Conv4 = nn.Conv2d(16, 1, 3, padding=1)
        self.BN4 = nn.BatchNorm2d(1)

    def forward(self, x):
        single_input = False

    # 如果输入是 3D，就自动加 batch 维
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
            single_input = True
        x = F.relu(self.BN1(self.Conv1(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.BN2(self.Conv2(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.BN3(self.Conv3(x)))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.BN4(self.Conv4(x)))
        x = F.max_pool2d(x, (2, 2))
        # 如果原来是单个样本，就去掉 batch 维
        if single_input:
            x = x.squeeze(0)

        return torch.sigmoid(x)

    def getImage(self, s, batch_size=1):
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

        if batch_size != 1:
            x = torch.randn(batch_size, 3, 28 * (2**4), 28 * (2**4)).to(next(self.parameters()).device)
        else:
            x = torch.randn(3, 28 * (2**4), 28 * (2**4)).to(next(self.parameters()).device)
        y = self.forward(x)
        
        # 回归原状态
        if preState == True:
            self.train()
        else:
            self.eval()

        return y

