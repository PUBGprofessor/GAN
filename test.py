from model import *
import os
import torch
from torchvision.utils import save_image
from Dataset import *
from torch.utils.data import DataLoader

class config:
    batch_size = 5
    epoch = 5
    version = "v3.1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = G().to(device)
D = D().to(device)
save_dir = os.path.join('./saved_models', config.version)

model_path_G = os.path.join(save_dir, f'G_epoch_{config.epoch}.pth')
model_path_D = os.path.join(save_dir, f'D_epoch_{config.epoch}.pth')
G.load_state_dict(torch.load(model_path_G))
D.load_state_dict(torch.load(model_path_D))
G.eval()    
D.eval()

# output_dir = './output_images'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# Imgs = G.getImage("test", batch_size=5)
# for ind, i in enumerate(Imgs):
#     save_image(i, os.path.join(output_dir, "test{}.png".format(ind)), normalize=True)

dataset = Dataset(G, device)
datasetLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

for i in datasetLoader:
    print(D.predict(G.getImage("test", 5)))
    print(D.predict(i[0]))
    break