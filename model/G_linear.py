import torch
import torch.nn as nn
import torch.nn.functional as F

    
# class G(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Conv1 = nn.Conv2d(3, 8, 3, padding=1)
#         self.BN1 = nn.BatchNorm2d(8)

#         self.Conv2 = nn.Conv2d(8, 32, 3, padding=1)
#         self.BN2 = nn.BatchNorm2d(32)

#         self.Conv3 = nn.Conv2d(32, 16, 3, padding=1)
#         self.BN3 = nn.BatchNorm2d(16)

#         self.Conv4 = nn.Conv2d(16, 1, 3, padding=1)
#         self.BN4 = nn.BatchNorm2d(1)

#     def forward(self, x):
#         single_input = False

#     # 如果输入是 3D，就自动加 batch 维
#         if x.dim() == 3:
#             x = x.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
#             single_input = True
#         x = F.relu(self.BN1(self.Conv1(x)))
#         x = F.max_pool2d(x, (2, 2))

#         x = F.relu(self.BN2(self.Conv2(x)))
#         x = F.max_pool2d(x, (2, 2))

#         x = F.relu(self.BN3(self.Conv3(x)))
#         x = F.max_pool2d(x, (2, 2))

#         x = F.relu(self.BN4(self.Conv4(x)))
#         x = F.max_pool2d(x, (2, 2))
#         # 如果原来是单个样本，就去掉 batch 维
#         if single_input:
#             x = x.squeeze(0)

#         return torch.sigmoid(x)

#     def getImage(self, s, batch_size=1):
#         if s not in ["train", "test"]:
#             raise ValueError("getImage:trian or test")
        
#         # 记录原状态
#         if self.training:
#             preState = True
#         else:
#             preState = False
        
#         if s == "train":
#             self.train()
#         else:
#             self.eval()

#         if batch_size != 1:
#             x = torch.randn(batch_size, 3, 28 * (2**4), 28 * (2**4)).to(next(self.parameters()).device)
#         else:
#             x = torch.randn(3, 28 * (2**4), 28 * (2**4)).to(next(self.parameters()).device)
#         y = self.forward(x)
        
#         # 回归原状态
#         if preState == True:
#             self.train()
#         else:
#             self.eval()

#         return y

class G(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 256)
        self.BN1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)  # 加入Dropout

        self.fc2 = nn.Linear(256, 512)
        self.BN2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)  # 加入Dropout

        self.fc3 = nn.Linear(512, 1024)
        self.BN3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.3)  # 加入Dropout

        self.fc4 = nn.Linear(1024, 1 * 28 * 28)
        self.BN4 = nn.BatchNorm1d(1 * 28 * 28)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        batch = x.size(0)

        x = self.dropout1(self.act(self.BN1(self.fc1(x))))
        x = self.dropout2(self.act(self.BN2(self.fc2(x))))
        x = self.dropout3(self.act(self.BN3(self.fc3(x))))
        x = self.act(self.BN4(self.fc4(x)))

        x = x.view(batch, 1, 28, 28)
        return torch.sigmoid(x) # 图片大小为(1, 28, 28) 取值范围[0, 1]

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

        x = torch.randn(batch_size, self.input_size).to(next(self.parameters()).device)

        y = self.forward(x)
        
        # 回归原状态
        if preState == True:
            self.train()
        else:
            self.eval()

        return y