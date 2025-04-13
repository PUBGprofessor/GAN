from model.G_linear import *
from model.D import D
import os
import torch
from torchvision.utils import save_image
from utils.image_utils import save_image_grid


class config:
    batch_size = 64
    epoch = 15
    version = "v3.3"
    fake_label = 0.95 # 假标签
    real_label = 0.05 # 假标签
    change_label = 0.2 # 交换标签的概率


save_dir = os.path.join('./saved_models', config.version)
output_dir = os.path.join('./output_images', config.version + "_epoch" +str(config.epoch))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = G().to(device)
model_path_G = os.path.join(save_dir, f'G_epoch_{config.epoch}.pth')
G.load_state_dict(torch.load(model_path_G,weights_only=True))
G.eval()



Imgs = G.getImage("test", batch_size=config.batch_size)
save_image_grid(Imgs, os.path.join(output_dir, "test.png"), nrow=8)
# for ind, i in enumerate(Imgs):
    # save_image(i, os.path.join(output_dir, "test{}.png".format(ind)), normalize=True)