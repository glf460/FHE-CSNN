
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from spikingjelly import visualizing
import numpy as np
from torchvision import transforms


# 对数据进行4级离散化的处理
class DiscretizeTransform:
    def __init__(self, levels=4):
        self.levels = levels

    def __call__(self, img):
        # 如果 img 不是张量，转换为张量
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        else:
            img = img.clone().detach().float()  # 推荐用法，避免警告

        # img = img /255.0
        # 进行离散化
        step = 1.0 / self.levels
        img = (img / step).floor()
        return img


train_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(),      # 随机水平翻转
        # transforms.RandomRotation(degrees=30),  # 随机旋转
        transforms.ToTensor(),  # 转换为Tensor
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
        DiscretizeTransform(levels=4)
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 测试集归一化
    DiscretizeTransform(levels=4)
])

class CSNN(nn.Module):
    def __init__(self, T, channels, tau, classes, use_cupy=False):
        super().__init__()
        self.T = T

        self.conv_fc = nn.Sequential(
            # 第一个卷积层1*28*28 -> 10*28*28
            layer.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
            # 经过第一个LIF 层，输出为 10*28*28
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            # LIFNode中，第二个参数为decay_input: bool = True，默认为 true
            # 经过池化层 10*28*28 -> 10*14*14
            layer.AvgPool2d(2, 2),

            # # 加入第二个卷积层10*14*14 -> 10*14*14
            # layer.Conv2d(10, channels, kernel_size=3, stride=1, padding=1, bias=False),
            # # 经过第一个LIF 层，输出为 10*14*14
            # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),  # LIFNode中，第二个参数为decay_input: bool = True，默认为 true
            # # 经过池化层 10*14*14 -> 10*7*7
            # layer.AvgPool2d(2, 2),

            # 拉平为10*7*7=490
            layer.Flatten(),
            # 经过一个线性层 490 -> 10*4*4
            layer.Linear(channels * 14 * 14, channels * 4 * 4, bias=False),  # channels = 10   channels * 8 * 8
            # 经过第二个LIF 层，输出为 10*4*4
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, classes, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

        if use_cupy:  # 选择后端加速
            functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

    # spiking_encoder 方法返回一个新的序列，包含了前 3 个层（卷积、批归一化和激活），通常用于生成脉冲编码器（spiking encoder），该脉冲编码器的作用是处理输入数据，将其转换为脉冲序列
    def spiking_encoder(self):
        return self.conv_fc[0:3]


def save_model_parameters(net, save_dir, tau):
    # 创建保存模型参数的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fmt = '%.18f'

    # 第一个卷积层权重
    conv1_weight = net.conv_fc[0].weight.detach().cpu().numpy()
    conv_weight_2d = conv1_weight.reshape(conv1_weight.shape[0], -1)
    np.savetxt(os.path.join(save_dir, f'conv1_weight_tau({tau}).txt'), conv_weight_2d, fmt=fmt)

    # 第一个线性层权重
    fc1_weight = net.conv_fc[4].weight.detach().cpu().numpy().T  # 转置为 (in_features, out_features)
    np.savetxt(os.path.join(save_dir, f'fc1_weight_tau({tau}).txt'), fc1_weight, fmt=fmt)

    # 第二个线性层权重
    fc2_weight = net.conv_fc[6].weight.detach().cpu().numpy().T  # 转置为 (in_features, out_features)
    np.savetxt(os.path.join(save_dir, f'fc2_weight_tau({tau}).txt'), fc2_weight, fmt=fmt)


def plot_metrics(train_metric, test_metric, metric_name, save_dir, dataset_name, tau_value):
    plt.figure()
    epochs = range(1, len(train_metric) + 1)

    # 绘制训练和测试曲线
    plt.plot(epochs, train_metric, color='#4682B4', label=f'Training {metric_name}', linewidth=1)
    plt.plot(epochs, test_metric, color='#32CD32', label=f'Testing {metric_name}', linewidth=1)

    plt.title(f'Training {metric_name} Curves on {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)

    plt.legend(loc='best')
    plt.grid(False)  # 移除网格线

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存图像
    save_path = os.path.join(save_dir, f'{dataset_name}_{metric_name.lower()}_curve_{tau_value}.png')
    plt.savefig(save_path, bbox_inches='tight')  # 确保图片不被裁剪
    plt.show()
    print(f"{metric_name} curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Classify')  #
    parser.add_argument('-dataset-name', default='MNIST', type=str, help='the name of dataset')  #
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-classes', default=10, type=int, help='number of classes')
    parser.add_argument('-b', default=256, type=int, help='batch size')
    parser.add_argument('-epochs', default=500, type=int, metavar='N', help='number of total epochs to run')  # 150
    parser.add_argument('-j', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='../../../../dataset/', help='root dir of dataset')
    parser.add_argument('-out-dir', type=str, default='../../../../logs/MNIST/1Conv2Linear/T4', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, default='Adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-channels', default=10, type=int, help='channels of CSNN')
    parser.add_argument('-save-es', default='../../../../save_img/MNIST/', help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-LIFNode}')
    parser.add_argument('-tau', default=4.0, type=float, help='parameter tau of LIF neuron')

    args = parser.parse_args()
    print(args)

    # 初始化模型
    net = CSNN(T=args.T, channels=args.channels, tau=args.tau, classes=args.classes, use_cupy=args.cupy)
    print(net)
    net.to(args.device)

    train_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=train_transforms,
        download=True
    )

    test_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=test_transforms,
        download=True
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,  # 在最后一个 batch 中，如果剩余的数据样本数量不足一个完整的 batch（即小于 batch_size），是否丢弃这些不完整的样本
        num_workers=args.j,
        pin_memory=True
    )

    # 初始化混合精度训练（如果启用）
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    # 选择优化器
    optimizer = None

    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # # 如果有断点恢复
    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     net.load_state_dict(checkpoint['net'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     max_test_acc = checkpoint['max_test_acc']
    #     if args.save_es is not None and args.save_es != '':
    #         encoder = net.spiking_encoder()
    #         with torch.no_grad():
    #             for img, label in test_data_loader:
    #                 img = img.to(args.device)
    #                 label = label.to(args.device)
    #                 # img.shape = [N, C, H, W]
    #                 img_seq = img.unsqueeze(0).repeat(net.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
    #                 spike_seq = encoder(img_seq)
    #                 functional.reset_net(encoder)
    #                 to_pil_img = torchvision.transforms.ToPILImage()
    #                 vs_dir = os.path.join(args.save_es, 'visualization')
    #                 os.mkdir(vs_dir)
    #
    #                 img = img.cpu()
    #                 spike_seq = spike_seq.cpu()
    #
    #                 # 28 * 28 is too small to read. So, we interpolate it to a larger size
    #                 # img = F.interpolate(img, scale_factor=4, mode='bilinear')  # 表示对图像放大4倍
    #
    #                 for i in range(label.shape[0]):
    #                     vs_dir_i = os.path.join(vs_dir, f'{i}')
    #                     os.mkdir(vs_dir_i)
    #                     to_pil_img(img[i]).save(os.path.join(vs_dir_i, f'input.png'))
    #
    #                     for t in range(net.T):
    #                         print(f'saving {i}-th sample with t={t}...')
    #                         # spike_seq.shape = [T, N, C, H, W]
    #
    #                         visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
    #
    #                         plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'), pad_inches=0.02)
    #                         plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'), pad_inches=0.02)
    #                         plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'), pad_inches=0.02)
    #                         plt.clf()  # 清空当前图像，以便为下一张图像做好准备
    #
    #                 exit()

    # 设置输出目录
    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}_tau{args.tau}')

    if args.amp:
        out_dir += '_amp'

    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # 初始化TensorBoard记录器
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    # 初始化存储列表
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, args.classes).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(img)
                    loss = F.mse_loss(out_fr, label_onehot)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # 记录到TensorBoard
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        # 记录到列表
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 学习率调度
        lr_scheduler.step()

        # 评估阶段
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, args.classes).float()
                out_fr = net(img)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples

        # 记录到TensorBoard
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        # 记录到列表
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 保存最佳模型
        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            # 保存最佳模型
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            # 保存最佳模型的参数
            save_model_parameters(net, f'../../../../save_model_parameter/MNIST/1Conv2Linear/T{args.T}', args.tau)

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        # 打印日志
        print(args)
        print(out_dir)
        print(f'epoch = {epoch + 1}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 关闭TensorBoard记录器
    writer.close()

    # 绘制并保存精度曲线
    acc_save_dir = os.path.join('../../../../save_img/MNIST/1Conv2Linear', f'acc/T{args.T}')
    plot_metrics(train_accuracies, test_accuracies, 'Accuracy', acc_save_dir, args.dataset_name, f'tau({args.tau})')

    # 绘制并保存损失曲线
    loss_save_dir = os.path.join('../../../../save_img/MNIST/1Conv2Linear', f'loss/T{args.T}')
    plot_metrics(train_losses, test_losses, 'Loss', loss_save_dir, args.dataset_name, f'tau({args.tau})')


if __name__ == '__main__':
    main()

