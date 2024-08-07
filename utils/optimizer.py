import torch.optim as optim


def get_optimizer(optimizer_type: str, optim_params, lr: float, momentum: float = 0.9, wd: float = 0.0):
    optimizer_type = optimizer_type.strip().lower()
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(optim_params, lr=lr, momentum=momentum, weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(optim_params, lr=lr, weight_decay=wd)
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(optim_params, lr=lr, weight_decay=wd)
    else:
        raise NotImplementedError(f'please provide correct optimizer name! <{optimizer_type}> not supported.')

    return optimizer
