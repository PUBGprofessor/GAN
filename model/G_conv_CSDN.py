import torch
import torch.nn as nn

# 生成器,基于上采样
class G_net_conv(nn.Module):
    def __init__(self, input_size=256):
        super(G_net_conv, self).__init__()
        self.input_size = input_size
        # 扩张数据量
        self.expand = nn.Sequential(
            nn.Linear(self.input_size, 484),
            nn.BatchNorm1d(484),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(484, 484),
            nn.BatchNorm1d(484),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
        )
        self.gen = nn.Sequential(
            # 反卷积扩张尺寸
            nn.ConvTranspose2d(1, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(4, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            # 1x1卷积压缩维度
            nn.Conv2d(4, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
            # 将输出约束到[-1,1]
            nn.Sigmoid()
        )

    def forward(self, img_seeds):
        img_seeds = self.expand(img_seeds)
        # 将线性数据重组为二维图片
        img_seeds = img_seeds.view(-1, 1, 22, 22)
        output = self.gen(img_seeds)
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