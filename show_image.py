# from model.G_linear import *
import os
import torch
from utils.model_utils import get_G_model, get_D_model

# from torchvision.utils import save_image
from utils.image_utils import save_image_grid


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

save_dir = os.path.join('./saved_models', config.version)
output_dir = os.path.join('./output_images', config.version + "_epoch" +str(config.epoch))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = get_G_model(config.G_model, device, bool(config.load_epoch), config.G_load_path)

G.eval()

Imgs = G.getImage("test", batch_size=config.batch_size)
save_image_grid(Imgs, os.path.join(output_dir, "test.png"), nrow=8)
# for ind, i in enumerate(Imgs):
    # save_image(i, os.path.join(output_dir, "test{}.png".format(ind)), normalize=True)