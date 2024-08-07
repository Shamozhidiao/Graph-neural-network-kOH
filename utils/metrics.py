import torch


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().cpu().squeeze()
    y_pred = y_pred.detach().cpu().squeeze()

    SS_res = torch.sum((y_true - y_pred) ** 2)
    SS_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - SS_res / SS_tot

    return r2.item()


def rmse_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().cpu().squeeze()
    y_pred = y_pred.detach().cpu().squeeze()

    mse = torch.mean((y_pred - y_true) ** 2)
    return torch.sqrt(mse)


def mae_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.detach().cpu().squeeze()
    y_pred = y_pred.detach().cpu().squeeze()

    return torch.mean(torch.abs(y_pred - y_true))


if __name__ == '__main__':
    y_true = torch.tensor([3, -0.5, 2, 7])
    y_pred = torch.tensor([2.5, 0.0, 2, 8])

    r2 = r2_score(y_true, y_pred)
    print("R2 score:", r2)
