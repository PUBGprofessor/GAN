import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from Dataset import *
import os

# 创建存储目录，如果不存在的话
save_dir = './saved_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = G().to(device)
D = D().to(device)

# def lossG(y1):
#     return torch.sum(1 - y1**2) / y1.shape[0]

def lossG(y1):
    return torch.sum(- torch.log(y1)) / y1.shape[0]

def lossD(y1, y2):
    y2 = y2.view(-1, 1)
    return F.mse_loss(y1, y2)

batch = 64 # 256
dataset = Dataset(G, device)
datasetLoader = DataLoader(dataset, batch_size=batch, shuffle=True)


Goptimer = optim.Adam(G.parameters(), 0.0005)
Doptimer = optim.Adam(D.parameters(), 0.005)

# for i in datasetLoader:
#     print(D.predict(G.getImage("test")))
#     break

epoch = 100

for j in range(epoch):
    for idx, i in enumerate(datasetLoader):
        Goptimer.zero_grad()
        Doptimer.zero_grad()

        # 训练D
        y = D.predict(i[0]) # 真标签
        loss1 = lossD(y, i[1])
        y = D.predict(G.getImage("test", batch)) # 假标签
        loss1 += lossD(y, torch.zeros([batch, 1], device=device))
        loss1.backward()
        Doptimer.step()
        Doptimer.zero_grad()

        # 训练G
        for j in range(200):
            Goptimer.zero_grad()
            x = G.getImage("train", batch * 2)
            y = D.predict(x)
            loss2 = lossG(y)
            loss2.backward()
            Goptimer.step()

        print('idx: {} loss1: {} loss2: {}'.format(idx, loss1, loss2))

    with torch.no_grad():
        print('epoch: {} loss1: {} loss2: {}'.format(j, loss1, loss2))
        # 每个epoch结束后保存模型
        model_path_G = os.path.join(save_dir, f'G_epoch_{j+1}.pth')
        model_path_D = os.path.join(save_dir, f'D_epoch_{j+1}.pth')
        torch.save(G.state_dict(), model_path_G)
        torch.save(D.state_dict(), model_path_D)

