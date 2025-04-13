import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.model_utils import get_G_model, get_D_model
from Dataset import *
from utils.image_utils import save_image_grid
import os
from utils.loss import lossD, lossG

class config:
    batch_size = 128
    epoch = 15
    version = "v5.0"
    # fake_label = 0.95 # 假标签
    # real_label = 0.05 # 假标签
    change_label = 0.1 # 交换标签的概率
    G_model = "G_linear_CSDN"
    D_model = "D_CSDN"
    load_epoch = 0
    G_load_path = os.path.join('./saved_models', version, f'G_epoch_{G_model}_{load_epoch}.pth')
    D_load_path = os.path.join('./saved_models', version, f'D_epoch_{D_model}_{load_epoch}.pth')

# 创建存储目录，如果不存在的话
save_dir = os.path.join('./saved_models', config.version)
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
# 创建输出目录
output_dir = os.path.join('./output_images', config.version)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据配置加载模型
G = get_G_model(config.G_model, device, bool(config.load_epoch), config.G_load_path)
D = get_D_model(config.D_model, device, bool(config.load_epoch), config.D_load_path)

G.train()
D.train()

batch = config.batch_size # 256

dataset = Dataset(device)
datasetLoader = DataLoader(dataset, batch_size=batch, shuffle=True)


Goptimer = optim.Adam(G.parameters(), 1e-4, weight_decay=1e-6) 
Doptimer = optim.Adam(D.parameters(), 1e-4, weight_decay=1e-6)

# for i in datasetLoader:
#     print(D.predict(G.getImage("test")))
#     break

epoch = config.epoch

for j in range(config.load_epoch, epoch):
    for idx, i in enumerate(datasetLoader):
        if i[0].shape[0] != config.batch_size: # 样本不足则跳过
            continue

        real_label = (1 - torch.rand(config.batch_size, 1)/10).to(device) # 0.9 ~ 1为真标签标签
        fake_label = torch.full([batch, 1], 0.0, device=device) # 假标签全为0

        # 训练D
        if j > config.epoch / 2 or torch.rand(1).item() > config.change_label * (config.epoch - j) / config.epoch: # 随迭代次数下调交换概率
            # 不交换标签
            y = D.predict(i[0], "train")
            D_loss = lossD(y, real_label) 
            y = D.predict(G.getImage("test", batch), "train") 
            y2 = fake_label # 假标签
        else: # 交换标签
            y = D.predict(i[0], "train")
            D_loss = lossD(y, i[1] * 0.0)
            y = D.predict(G.getImage("test", batch), "train")
            y2 = real_label
        D_loss += lossD(y, y2)
        D_loss = D_loss / 2

        Doptimer.zero_grad()
        D_loss.backward()
        Doptimer.step()

        # 训练G
        for g_step in range(1):
            x = G.getImage("train", batch)
            y = D.predict(x, "train")
            # y = D(x)
            y2 = real_label # 真实标签
            loss2 = lossG(y, y2)
            Goptimer.zero_grad()
            loss2.backward()
            Goptimer.step()
        
        if idx % 100 == 0:
            print('idx: {} D_loss: {} loss2: {}'.format(idx, D_loss, loss2))

    with torch.no_grad():
        print("####################################")
        print('epoch: {} D_loss: {} loss2: {}'.format(j+1, D_loss, loss2))
        print("####################################")
        # 每个epoch结束后保存模型
        model_path_G = os.path.join(save_dir, f'G_epoch_{config.D_model}_{j+1}.pth')
        torch.save(G.state_dict(), model_path_G)
        if (j+1) % 5 == 0:
            model_path_D = os.path.join(save_dir, f'D_epoch_{config.G_model}_{j+1}.pth')
            torch.save(D.state_dict(), model_path_D)

        Imgs = G.getImage("test", batch_size=config.batch_size)
        save_image_grid(Imgs, os.path.join(output_dir, f"epoch{j + 1}.png"), nrow=8)

