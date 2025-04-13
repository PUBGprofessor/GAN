

import torch
import torch.nn as nn


# 生成器，基于线性层
class G_net_linear(nn.Module):
    def __init__(self, input_size=256):
        super(G_net_linear, self).__init__()
        self.input_size = input_size
        self.gen = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            # 将输出约束到[-1,1]
            nn.Sigmoid()
        )

    def forward(self, img_seeds):
        output = self.gen(img_seeds)
        # 将线性数据重组为二维图片
        output = output.view(-1, 1, 28, 28)
        return output

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