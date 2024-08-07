# from dataset.molecule_dataset import MoleculeDatasetWrapper
from dataset.new_molecule_dataset import MoleculeDatasetWrapper
from models.gcn import GCN
from models.gin_v1 import GIN as GIN_v1
from models.gin_v2 import GIN as GIN_v2

from utils.tools import set_deterministic, plot_custom_figure
from utils.tools import Timer, AverageMeter, ProgressMeter
from utils.optimizer import get_optimizer
from utils.scheduler import adjust_learning_rate
from utils.loss import get_loss_function
from utils.metrics import r2_score, rmse_score, mae_score

import warnings
warnings.filterwarnings("ignore", message='not removing hydrogen atom without neighbors')

import os
import os.path as osp
import shutil
import time
import argparse
import functools
import pdb

import torch


def print_(s: str, filepath=None):
    # print on screen (and into file)
    print(s)
    if filepath:
        with open(filepath, 'a+') as f:
            print(s, file=f)


def main(args):
    # record experiment time
    exp_timer = Timer()
    exp_timer.begin()

    # for reproducility
    set_deterministic(args.seed)

    # training device
    args.device = f'cuda:{args.gpu}'
    torch.cuda.set_device(args.gpu)

    # dataset and dataloader
    dataset = MoleculeDatasetWrapper(dataset_path=args.ds_path, batch_size=args.batch_size,
                                     num_workers=args.workers, record_path=args.split_ds_dir, split=args.split)
    train_loader, val_loader, test_loader = dataset.get_train_val_test_data_loaders()


    # model
    if args.model_name == 'gcn':
        model = GCN(num_layer=args.num_layer, emb_dim=args.emb_dim, feat_dim=args.feat_dim,
                    drop_ratio=args.drop_ratio, pred_n_layer=args.pred_n_layer, pred_act=args.pred_act)
    elif args.model_name == 'gin_v1':
        model = GIN_v1(num_layer=args.num_layer, emb_dim=args.emb_dim,
                       drop_ratio=args.drop_ratio, pred_n_layer=args.pred_n_layer, pred_act=args.pred_act)
    elif args.model_name == 'gin_v2':
        model = GIN_v2(num_layer=args.num_layer, emb_dim=args.emb_dim, feat_dim=args.feat_dim,
                       drop_ratio=args.drop_ratio, pred_n_layer=args.pred_n_layer, pred_act=args.pred_act)
    else:
        raise NotImplementedError('please provide right model name!')

    if args.backbone_weights:
        state_dict = torch.load(args.backbone_weights, map_location='cpu')
        model.load_my_state_dict(state_dict)
    model.cuda()

    # print model
    print_(model, args.train_log_path)
    print_('-' * 50, args.train_log_path)

    # # optimizer
    # optim_params = model.parameters()
    # optimizer = get_optimizer(optimizer_type=args.optimizer_type,
    #                           optim_params=optim_params,
    #                           lr=args.lr, momentum=args.momentum, wd=args.wd)
    # # lr scheduler
    # adjust_lr = functools.partial(adjust_learning_rate,
    #                               scheduler_type=args.scheduler_type,
    #                               steps=args.steps, gamma=args.gamma,
    #                               warmup_epochs=args.warmup_epochs, epochs=args.epochs,
    #                               lr=args.lr, min_lr=args.min_lr)

    # optimizer
    head_params_list = []
    for name, param in model.named_parameters():
        if 'pred_head' in name:
            head_params_list.append(name)

    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in head_params_list, model.named_parameters()))))
    head_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in head_params_list, model.named_parameters()))))

    # optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.base_lr}, {'params': head_params}],
    #                              lr=args.lr, weight_decay=args.wd)

    base_optimizer = get_optimizer(optimizer_type=args.optimizer_type,
                                   optim_params=base_params, lr=args.base_lr, momentum=args.momentum, wd=args.wd)
    head_optimizer = get_optimizer(optimizer_type=args.optimizer_type,
                                   optim_params=head_params, lr=args.lr, momentum=args.momentum, wd=args.wd)
    base_adjust_lr = functools.partial(adjust_learning_rate,
                                       scheduler_type=args.scheduler_type, steps=args.steps, gamma=args.gamma,
                                       warmup_epochs=args.warmup_epochs, epochs=args.epochs,
                                       lr=args.base_lr, min_lr=args.min_lr)
    head_adjust_lr = functools.partial(adjust_learning_rate,
                                       scheduler_type=args.scheduler_type, steps=args.steps, gamma=args.gamma,
                                       warmup_epochs=args.warmup_epochs, epochs=args.epochs,
                                       lr=args.lr, min_lr=args.min_lr)

    # pdb.set_trace()

    # loss
    loss_fn = get_loss_function(loss_type=args.loss_type)
    loss_fn = loss_fn.cuda()

    train_losses = []
    train_r2s, val_r2s, test_r2s = [], [], []
    train_rmse_list, val_rmse_list, test_rmse_list = [], [], []
    train_mae_list, val_mae_list, test_mae_list = [], [], []
    best_r2, best_mae, best_rmse, best_epoch = float('-inf'), float('inf'), float('inf'), 0

    epoch_timer = Timer()  # record epoch time
    for epoch in range(args.epochs):
        epoch_timer.begin()

        cur_base_lr = base_adjust_lr(epoch, base_optimizer)
        cur_head_lr = head_adjust_lr(epoch, head_optimizer)
        print_('current base learning rate: {:.8f}'.format(cur_base_lr), args.train_log_path)
        print_('current head learning rate: {:.8f}'.format(cur_head_lr), args.train_log_path)

        torch.cuda.empty_cache()

        # train
        print_('training...', args.train_log_path)
        train_loss = train_epoch(train_loader, model, base_optimizer, head_optimizer, loss_fn, epoch + 1, args)
        train_losses.append(train_loss)

        # validation
        print_('validating...', args.train_log_path)
        train_r2, train_rmse, train_mae = validate(train_loader, model, epoch + 1, args, 'Train')
        val_r2, val_rmse, val_mae = validate(val_loader, model, epoch + 1, args, 'Val')
        test_r2, test_rmse, test_mae = validate(test_loader, model, epoch + 1, args, 'Test')

        train_r2s.append(train_r2), val_r2s.append(val_r2), test_r2s.append(test_r2)
        train_rmse_list.append(train_rmse), val_rmse_list.append(val_rmse), test_rmse_list.append(test_rmse)
        train_mae_list.append(train_mae), val_mae_list.append(val_mae), test_mae_list.append(test_mae)

        torch.cuda.empty_cache()

        # plot figure about train loss
        train_loss_dict = {'train_loss': train_losses}
        plot_custom_figure(epoch + 1, train_loss_dict, 'train_loss', args.figure_dir)

        # plot figure about evaluation metrics
        r2_metrics_dict = {'train_r2': train_r2s, 'val_r2': val_r2s, 'test_r2': test_r2s}
        plot_custom_figure(epoch + 1, r2_metrics_dict, 'train_val_test_r2_scores', args.figure_dir, y_lim=(-0.5, 1.0))

        rmse_metrics_dict = {'train_rmse': train_rmse_list, 'val_rmse': val_rmse_list, 'test_rmse': test_rmse_list}
        plot_custom_figure(epoch + 1, rmse_metrics_dict, 'train_val_test_rmse_scores', args.figure_dir, y_lim=(.0, 1.0))

        mae_metrics_dict = {'train_mae': train_mae_list, 'val_mae': val_mae_list, 'test_mae': test_mae_list}
        plot_custom_figure(epoch + 1, mae_metrics_dict, 'train_val_test_mae_scores', args.figure_dir, y_lim=(.0, 1.0))

        if args.val_metric == 'r2' and val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            ckpt = {'state_dict': model.state_dict()}
            torch.save(ckpt, args.best_model_weights)

        if args.val_metric == 'mae' and val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            ckpt = {'state_dict': model.state_dict()}
            torch.save(ckpt, args.best_model_weights)

        if args.val_metric == 'rmse' and val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            ckpt = {'state_dict': model.state_dict()}
            torch.save(ckpt, args.best_model_weights)

        epoch_timer.end()
        print_(epoch_timer.time_str(), args.train_log_path)
        print_('-' * 50, args.train_log_path)

    print_(f'Best Epoch: {best_epoch}', args.train_log_path)
    print_(
        f'Final Train R2 Score: {train_r2s[best_epoch - 1]:.4f}, Final Train RMSE: {train_rmse_list[best_epoch - 1]:.4f}, Final Train MAE: {train_mae_list[best_epoch - 1]:.4f}', args.train_log_path)
    print_(
        f'Final Val R2 Score: {val_r2s[best_epoch - 1]:.4f}, Final Val RMSE: {val_rmse_list[best_epoch - 1]:.4f}, Final Val MAE: {val_mae_list[best_epoch - 1]:.4f}', args.train_log_path)
    print_(
        f'Final Test R2 Score: {test_r2s[best_epoch - 1]:.4f}, Final Test RMSE: {test_rmse_list[best_epoch - 1]:.4f}, Final Test MAE: {test_mae_list[best_epoch - 1]:.4f}', args.train_log_path)
    ckpt = {'state_dict': model.state_dict()}
    torch.save(ckpt, args.last_model_weights)

    exp_timer.end()
    print_(exp_timer.time_str(), args.train_log_path)


def train_epoch(loader, model, base_optimizer, head_optimizer, loss_fn, epoch, args):
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses],
        prefix='Epoch [{}/{}]'.format(epoch, args.epochs)
    )

    model.train()

    end = time.time()
    for batch, (data, log_k) in enumerate(loader):
        data_time.update(time.time() - end)  # measure data loading time
        data = data.cuda()
        log_k = log_k.cuda()
        _, y_hat, a = model(data)
        loss = loss_fn(log_k.squeeze(), y_hat.squeeze())  # forward
        losses.update(loss.item(), log_k.size(0))  # record loss
        # backward
        base_optimizer.zero_grad()
        head_optimizer.zero_grad()
        loss.backward()
        base_optimizer.step()
        head_optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % args.print_freq == 0:
            print_(progress.display(batch), args.train_log_path)

    return losses.avg


def validate(loader, model, epoch, args, loader_type='Val'):
    model.eval()

    pred_list = []
    true_list = []
    for data, log_k in loader:
        data = data.cuda()
        _, y_hat,a = model(data)
        pred_list.append(y_hat.detach().cpu())
        true_list.append(log_k.detach().cpu())
    pred_tensor = torch.cat(pred_list, dim=0)
    true_tensor = torch.cat(true_list, dim=0)

    r2 = r2_score(y_true=true_tensor, y_pred=pred_tensor)
    rmse = rmse_score(y_true=true_tensor, y_pred=pred_tensor)
    mae = mae_score(y_true=true_tensor, y_pred=pred_tensor)
    print_('Epoch [{}/{}] {} R2 Score: {:.4f}, RMSE: {:.4f}, MAE: {:.4f}'.format(epoch,
           args.epochs, loader_type, r2, rmse, mae), args.train_log_path)

    return r2, rmse, mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    # --------------------------------------------------
    parser.add_argument('--exp_name', type=str, default='MolPredK')
    # data
    parser.add_argument('--ds_path', type=str, default='./new_molecule_dataset.csv', help='dataset path')
    parser.add_argument('--split', type=str, default='stratified',
                        choices=['random', 'scaffold', 'stratified', 'random_stratified'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    # model
    parser.add_argument('--model_name', type=str, default='gin_v1', choices=['gin_v1', 'gin_v2', 'gcn'])
    parser.add_argument('--backbone_weights', type=str,
                        default='./pretrain_ckpt/MoleBERT_gin.pth',
                        help='model weights, pretrained from self-supervised learning')
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--drop_ratio', type=float, default=0.1)
    parser.add_argument('--pred_n_layer', type=int, default=1)
    parser.add_argument('--pred_act', type=str, default='softplus', choices=['relu', 'softplus'])
    # training
    parser.add_argument('--optimizer_type', type=str, default='Adam', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--base_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument("--min_lr", type=float, default=1e-6, help="minimum learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--scheduler_type', type=str, default='constant', choices=['constant', 'cosine', 'step'])
    parser.add_argument('--steps', type=int, default=[100], nargs='+', help='only used for step scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='only used for step scheduler')
    parser.add_argument('--loss_type', type=str, default='charbonnier', choices=['l1', 'l2', 'charbonnier'])
    # validation
    parser.add_argument('--val_metric', type=str, default='mae', choices=['r2', 'mae', 'rmse'])
    # device
    parser.add_argument('--gpu', type=int, default=0)
    # log, figure and checkpoint
    parser.add_argument('--record_dir', type=str, default='./experiments/')
    # display
    parser.add_argument('--print_freq', type=int, default=1, help='frequency of printing info')
    # reproducibility
    parser.add_argument('--seed', type=int, default=1)
    # --------------------------------------------------
    args = parser.parse_args()

    if len(args.steps) == 1:
        args.steps = args.steps[0]

    # collect experiment info
    time_stamp = time.strftime('%Y-%m-%d-%H-%M')
    if args.seed is not None:
        info = f'{args.model_name}_e{args.epochs}_{args.optimizer_type}_{args.lr:.0e}_seed{args.seed}_{time_stamp}'
    else:
        info = f'{args.model_name}_e{args.epochs}_{args.optimizer_type}_{args.lr:.0e}_{time_stamp}'

    # record dir
    if not osp.exists(args.record_dir):
        os.mkdir(args.record_dir)
    args.record_dir = osp.join(args.record_dir, info + '/')
    if osp.exists(args.record_dir):
        shutil.rmtree(args.record_dir)
    os.mkdir(args.record_dir)

    # figure dir
    args.figure_dir = osp.join(args.record_dir, 'figure/')
    os.mkdir(args.figure_dir)

    # split dataset dir
    args.split_ds_dir = osp.join(args.record_dir, 'split_dataset/')
    os.mkdir(args.split_ds_dir)

    args.train_log_path = osp.join(args.record_dir, 'train_log.txt')
    args.best_model_weights = osp.join(args.record_dir, 'best_weights.pth')
    args.last_model_weights = osp.join(args.record_dir, 'last_weights.pth')

    exp_info = 'Experiment settings' + '\n' + '-' * 50 + '\n'  # experiment's settings
    for k, v in vars(args).items():
        exp_info += f'{k}: {v}\n'
    exp_info += '-' * 50
    args.exp_info = exp_info

    # record experiment settings
    print_(args.exp_info, args.train_log_path)

    main(args)
