import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.G_linear import G
from model.D import D
from Dataset import *
import os

class config:
    batch_size = 128
    epoch = 50
    version = "v4.0"
    # fake_label = 0.95 # 假标签
    # real_label = 0.05 # 假标签
    change_label = 0.1 # 交换标签的概率

    load_epoch = 0
    G_load_path = os.path.join('./saved_models', version, f'G_epoch_{load_epoch}.pth')
    D_load_path = os.path.join('./saved_models', version, f'D_epoch_{load_epoch}.pth')

# 创建存储目录，如果不存在的话
save_dir = os.path.join('./saved_models', config.version)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = G().to(device)
D = D().to(device)

if config.load_epoch != 0 and os.path.exists(config.G_load_path):
    G.load_state_dict(torch.load(config.G_load_path, weights_only=True))
    D.load_state_dict(torch.load(config.D_load_path, weights_only=True))
G.train()
D.train()
# def lossG(y1):
#     return torch.sum(1 - y1**2) / y1.shape[0]
# def lossG(y1):
#     return -torch.mean(torch.log(y1 + 1e-6))  # 防止 log(0)

criterion = torch.nn.BCELoss()
def lossG(y1):
    y2 = torch.full([y1.shape[0], 1], 1.0, device=device) # 0.95假标签
    return criterion(y1, y2)

# criterion = torch.nn.BCEWithLogitsLoss()
# def lossG(y1):
#     # y1 = y1.view(-1, 1)
#     y2 = torch.full([y1.shape[0], 1], 0.95, device=device) # 0.95假标签
#     return criterion(y1, y2)  # 使用 BCEWithLogitsLoss 计算损失

def lossD(y1, y2):
    y2 = y2.view(-1, 1)
    return criterion(y1, y2)

batch = config.batch_size # 256
dataset = Dataset(G, device)
datasetLoader = DataLoader(dataset, batch_size=batch, shuffle=True)


Goptimer = optim.Adam(G.parameters(), 1e-4, betas=(0.5, 0.999), weight_decay=1e-4) 
Doptimer = optim.Adam(D.parameters(), 1e-4, betas=(0.5, 0.999), weight_decay=1e-4)

# for i in datasetLoader:
#     print(D.predict(G.getImage("test")))
#     break

epoch = config.epoch

for j in range(config.load_epoch, epoch):
    for idx, i in enumerate(datasetLoader):
        if i[0].shape[0] != config.batch_size: # 样本不足则跳过
            continue
        Goptimer.zero_grad()
        Doptimer.zero_grad()

        real_label = (1 - torch.rand(config.batch_size, 1)/10).to(device)
        # 训练D
        if j > config.epoch / 2 or torch.rand(1).item() > config.change_label * (config.epoch - j) / config.epoch: # 随迭代次数下调交换概率
            # 不交换标签
            y = D.predict(i[0], "train")
            loss1 = lossD(y, real_label) # 0.9 ~ 1假标签
            y = D.predict(G.getImage("test", batch), "train") 
            y2 = torch.full([batch, 1], 0.0, device=device)
        else: # 交换标签
            y = D.predict(i[0], "train")
            loss1 = lossD(y, i[1] * 0.0)
            y = D.predict(G.getImage("test", batch), "train")
            y2 = real_label
        loss1 += lossD(y, y2)
        loss1 = loss1 / 2
        loss1.backward()
        Doptimer.step()
        Doptimer.zero_grad()

        # 训练G
        for g_step in range(1):
            x = G.getImage("train", batch)
            y = D.predict(x, "train")
            # y = D(x)
            loss2 = lossG(y)
            Goptimer.zero_grad()
            loss2.backward()
            Goptimer.step()
        if idx % 10 == 0:
            print('idx: {} loss1: {} loss2: {}'.format(idx, loss1, loss2))

    with torch.no_grad():
        print("####################################")
        print('epoch: {} loss1: {} loss2: {}'.format(j+1, loss1, loss2))
        print("####################################")
        # 每个epoch结束后保存模型
        if (j+1) % 5 == 0:
            model_path_G = os.path.join(save_dir, f'G_epoch_{j+1}.pth')
            model_path_D = os.path.join(save_dir, f'D_epoch_{j+1}.pth')
            torch.save(G.state_dict(), model_path_G)
            torch.save(D.state_dict(), model_path_D)

