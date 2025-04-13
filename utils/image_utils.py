import torchvision.utils as vutils
import matplotlib.pyplot as plt

def save_image_grid(images, save_path='output.png', nrow=8):
    """
    将多张图像拼接成一个网格并保存
    :param images: 形状为 (N, C, H, W) 的 tensor
    :param save_path: 保存路径
    :param nrow: 每行多少张图
    """
    # 把图像拼接成一个大图，normalize=True会自动将像素值缩放到[0,1]
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, padding=2)

    # 使用 matplotlib 保存
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # CHW -> HWC
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()