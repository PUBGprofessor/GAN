import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from Dataset import *
import os

class config:
    batch_size = 5
    epoch = 1
    version = "v3.0"

# 创建存储目录，如果不存在的话
save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = G().to(device)
D = D().to(device)

def lossG(y1):
    return torch.sum(1 - y1**2) / y1.shape[0]

def lossG(y1):
    return -torch.mean(torch.log(y1 + 1e-4))  # 防止 log(0)

# criterion = torch.nn.BCEWithLogitsLoss()
# def lossG(y1):
#     # y1 = y1.view(-1, 1)
#     y2 = torch.full([y1.shape[0], 1], 0.95, device=device) # 0.95假标签
#     return criterion(y1, y2)  # 使用 BCEWithLogitsLoss 计算损失

def lossD(y1, y2):
    y2 = y2.view(-1, 1)
    return F.mse_loss(y1, y2)

batch = 128 # 256
dataset = Dataset(G, device)
datasetLoader = DataLoader(dataset, batch_size=batch, shuffle=True)


Goptimer = optim.Adam(G.parameters(), 1e-4)
Doptimer = optim.Adam(D.parameters(), 2e-4)

# for i in datasetLoader:
#     print(D.predict(G.getImage("test")))
#     break

epoch = 5

for j in range(epoch):
    for idx, i in enumerate(datasetLoader):
        Goptimer.zero_grad()
        Doptimer.zero_grad()

        # 训练D
        y = D.predict(i[0])
        loss1 = lossD(y, i[1] * 0.95) # 0.95假标签
        y = D.predict(G.getImage("test", batch)) 
        y2 = torch.full([batch, 1], 0.05, device=device) # 0.05假标签
        loss1 += lossD(y, y2)
        loss1.backward()
        Doptimer.step()
        Doptimer.zero_grad()

        # 训练G
        for g_step in range(30):
            x = G.getImage("train", batch)
            y = D.predict(x)
            # y = D(x)
            loss2 = lossG(y)
            Goptimer.zero_grad()
            loss2.backward()
            Goptimer.step()
        if idx % 10 == 0:
            print('idx: {} loss1: {} loss2: {}'.format(idx, loss1, loss2))

    with torch.no_grad():
        print("####################################")
        print('epoch: {} loss1: {} loss2: {}'.format(j, loss1, loss2))
        print("####################################")
        # 每个epoch结束后保存模型
        model_path_G = os.path.join(save_dir, config.version, f'G_epoch_{j+1}.pth')
        model_path_D = os.path.join(save_dir, config.version, f'D_epoch_{j+1}.pth')
        torch.save(G.state_dict(), model_path_G)
        torch.save(D.state_dict(), model_path_D)

