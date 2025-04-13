import torch

# 根据生成器的配置返回对应的模型
def get_G_model(G_type, device, load_model=False, model_path=None):
    if G_type == "G_linear_CSDN":
        from model.G_linear_CSDN import G_net_linear
        model = G_net_linear()
    elif G_type == "G_conv_CSDN":
        from model.G_conv_CSDN import G_net_conv
        model = G_net_conv()
    elif G_type == "G_linear":
        from model.G_linear import G
        model = G()
    else:
        raise ValueError("Invalid G_type: {}".format(G_type))
    # 从磁盘加载之前保存的模型参数
    if model_path is not None and load_model:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model

# 返回判别器的模型
def get_D_model(D_type, device, load_model=False, model_path=None):
    if D_type == "D":
        from model.D import D
        model = D()
    elif D_type == "D_CSDN":
        from model.D_CSDN import D_net
        model = D_net()
    else:
        raise ValueError("Invalid D_type: {}".format(D_type))
    # 从磁盘加载之前保存的模型参数
    if model_path is not None and load_model:
        model.load_state_dict(torch.load(model_path))
    # 将模型加载到用于运算的设备的内存
    model = model.to(device)

    return model