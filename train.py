import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.model_utils import get_G_model, get_D_model
from Dataset import *
import os
from utils.loss import lossD, lossG
class config:
    batch_size = 128
    epoch = 50
    version = "v5.0"
    # fake_label = 0.95 # 假标签
    # real_label = 0.05 # 假标签
    change_label = 0.1 # 交换标签的概率
    G_model = "G_linear_CSDN"
    D_model = "D_CSDN"
    load_epoch = 0
    G_load_path = os.path.join('./saved_models', version, f'G_epoch_{load_epoch}.pth')
    D_load_path = os.path.join('./saved_models', version, f'D_epoch_{load_epoch}.pth')

# 创建存储目录，如果不存在的话
save_dir = os.path.join('./saved_models', config.version)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据配置加载模型
G = get_G_model(config.G_model, device, bool(config.load_epoch), config.G_load_path)
D = get_D_model(config.D_model, device, bool(config.load_epoch), config.D_load_path)

G.train()
D.train()

batch = config.batch_size # 256

dataset = Dataset(device)
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
            y2 = torch.full([y.shape[0], 1], 1.0, device=device)
            loss2 = lossG(y, y2)
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

