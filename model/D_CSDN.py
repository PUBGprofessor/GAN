import torch
import torch.nn as nn
# 判别器
class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(36864, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, img):
        # 提取特征
        features = self.features(img)
        # 展平二维矩阵
        features = features.view(features.shape[0],-1)
        # 使用线性层分类
        output = self.classifier(features)
        return output
    
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