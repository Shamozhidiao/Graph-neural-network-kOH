import math


def adjust_learning_rate(
    epoch, optimizer,
    scheduler_type='cosine',
    steps=None, gamma=None,
    warmup_epochs: int = 0, epochs=None, lr=None, min_lr=1e-6
):
    scheduler_type = scheduler_type.strip().lower()
    # epoch in [0, epochs - 1]
    if scheduler_type == 'constant':
        cur_lr = optimizer.param_groups[0]['lr']
    elif scheduler_type == 'step':  # step or multi step
        cur_lr = optimizer.param_groups[0]['lr']
        assert steps is not None and gamma is not None
        # steps: step size (int) or milestones (list or tuple)
        # gamma: decay factor (such as 0.5)
        if isinstance(steps, int):
            if epoch > 0 and epoch % steps == 0:
                cur_lr = cur_lr * gamma
        elif isinstance(steps, (list, tuple)):
            if epoch in steps:
                cur_lr = cur_lr * gamma
        else:
            raise TypeError('please provide right <steps> argumment')
    elif scheduler_type == 'cosine':  # cosine annealing with warmup
        assert epochs is not None and lr is not None
        # warmup_epochs: number of warmup training epochs
        # epochs: number of total training epochs
        # lr: base learning rate
        if epoch < warmup_epochs:
            cur_lr = min_lr + (lr - min_lr) * (epoch + 1) / warmup_epochs
        else:
            factor = math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))
            cur_lr = min_lr + (lr - min_lr) * 0.5 * (1.0 + factor)
    else:
        raise NotImplementedError(f'please provide right scheduler type!')

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    return cur_lr
