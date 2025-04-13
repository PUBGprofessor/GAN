import torch
# def lossG(y1):
#     return torch.sum(1 - y1**2) / y1.shape[0]
# def lossG(y1):
#     return -torch.mean(torch.log(y1 + 1e-6))  # 防止 log(0)

criterion = torch.nn.BCELoss()

def lossG(y1, y2):
    return criterion(y1, y2)

# criterion = torch.nn.BCEWithLogitsLoss()
# def lossG(y1):
#     # y1 = y1.view(-1, 1)
#     y2 = torch.full([y1.shape[0], 1], 0.95, device=device) # 0.95假标签
#     return criterion(y1, y2)  # 使用 BCEWithLogitsLoss 计算损失

def lossD(y1, y2):
    y2 = y2.view(-1, 1)
    return criterion(y1, y2)
