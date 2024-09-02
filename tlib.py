# pytorch 实用函数库
# Author: CJL
# 大部分函数实现出自[动手学深度学习(d2l)](https://zh.d2l.ai/)
# version: 0.2.0

## 计时器
import time
import numpy as np


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


## 累加器
class Accumulator:  # from d2l
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


## Logger
class Logger:
    """输出log"""

    def __init__(self, level="trace", output="stdout"):
        self.output = output
        self.level = level
        assert level in ["error", "warning", "info", "debug", "trace"]
        self.level_map = {"error": 5, "warning": 4, "info": 3, "debug": 2, "trace": 1}
        self.level_num = self.level_map[level]

    def get_log_text_color(self, level, text) -> str:
        if level == "error":
            return f"\x1b[31;1;4m[X] Error\x1b[0m\x1b[31m: {text}\x1b[0m"
        elif level == "warning":
            return f"\x1b[33;1;4m[!] Warning\x1b[0m\x1b[33m: {text}\x1b[0m"
        elif level == "info":
            return f"\x1b[34;1;4m[+] Info\x1b[0m\x1b[34m: {text}\x1b[0m"
        elif level == "debug":
            return f"\x1b[36;1;4m[#] Debug\x1b[0m\x1b[36m: {text}\x1b[0m"
        elif level == "trace":
            return f"{text}"
        else:
            return f"\x1b[31;1;4m[X] Unknown\x1b[0m\x1b[31m: {text}\x1b[0m"

    def get_log_text(self, level, text) -> str:
        if level == "error":
            return f"[X] Error: {text}"
        elif level == "warning":
            return f"[!] Warning: {text}"
        elif level == "info":
            return f"[+] Info: {text}"
        elif level == "debug":
            return f"[#] Debug: {text}"
        elif level == "trace":
            return f"{text}"
        else:
            return f"[X] Unknown: {text}"

    def print(self, level, text):
        if self.output == "stdout":
            print(self.get_log_text_color(level, text))
        else:
            with open(self.output, "a") as f:
                f.write(self.get_log_text(level, text) + "\n")

    def error(self, text):
        self.print("error", text)

    def warning(self, text):
        if self.level_num <= 4:
            self.print("warning", text)

    def info(self, text):
        if self.level_num <= 3:
            self.print("info", text)

    def debug(self, text):
        if self.level_num <= 2:
            self.print("debug", text)

    def trace(self, text):
        if self.level_num <= 1:
            self.print("trace", text)


## 绘图

### 静态图片

# 默认在 jupyter notebook 中绘图
# from d2l
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    xlabel=None,
    ylabel=None,
    legend=[],
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=("-", "m--", "g-.", "r:"),
    figsize=(3.5, 2.5),
    axes=None,
):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


### 动态图片
# 默认在 jupyter notebook 中绘图
# from d2l
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
import numpy as np


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """在动画中绘制数据"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        nrows=1,
        ncols=1,
        figsize=(3.5, 2.5),
    ):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [
                self.axes,
            ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


## checkpoint manager
import os, shutil


class CheckpointManager:
    """
    checkpoint standard:
    {
        'epoch': int,
        'arch': str,
        'dataset': str,
        'state_dict': dict,
        'prec': float,
        'total_batch': int,
    }
    """

    def __init__(self, arch, dataset, logger=None):
        self.arch = arch
        self.dataset = dataset
        self.isloaded = False
        self.best_prec = 0.0
        self.epoch = 0
        self.total_batch = 0
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger()

    def check_diff(self, arch, dataset, path):
        """
        检查checkpoint的架构和数据集是否匹配
        """
        if self.arch != arch:
            self.logger.warning(
                f"Arch {self.arch} does not match {arch} in checkpoint {path}."
            )
        if self.dataset != dataset:
            self.logger.warning(
                f"Dataset {self.dataset} does not match {dataset} in checkpoint {path}."
            )

    def parse_checkpoint(self, path, isbest=False, use_best=False):
        """
        读取checkpoint中的信息
        ```python
        cpm = CheckpointManager()
        checkpoint = cpm.parse_checkpoint(path)
        ```
        """
        if not os.path.exists(path):
            self.logger.warning(f"Checkpoint path {path} does not exist.")
            return None
        checkpoint = torch.load(path)
        if not isbest:
            self.logger.trace(
                f"Checkpoint: epoch={checkpoint['epoch']}, total_batch={checkpoint['total_batch']}, dataset={checkpoint['dataset']}, arch={checkpoint['arch']}, prec={checkpoint['prec']}"
            )
            self.check_diff(checkpoint["arch"], checkpoint["dataset"], path)
        else:
            self.logger.trace(
                f"Best Prec: {checkpoint['prec']}, best epoch={checkpoint['epoch']}"
            )
            self.check_diff(checkpoint["arch"], checkpoint["dataset"], path)
            self.best_prec = checkpoint["prec"]
        if use_best and isbest or not use_best and not isbest:
            self.epoch = checkpoint["epoch"]
            self.total_batch = checkpoint["total_batch"]
        return checkpoint["state_dict"]

    def load(self, model, checkpoint_path: str, best_path: str, use_best: bool = False):
        self.checkpoint_path = checkpoint_path
        self.best_path = best_path
        self.isloaded = True
        if not os.path.exists(checkpoint_path) or not os.path.exists(best_path):
            self.logger.info(
                f"Checkpoint path {checkpoint_path} or {best_path} does not exist. Checkpoint not loaded."
            )
            return model
        checkpoint_state_dict = self.parse_checkpoint(checkpoint_path, use_best)
        best_state_dict = self.parse_checkpoint(
            best_path, isbest=True, use_best=use_best
        )
        if use_best:
            model.load_state_dict(best_state_dict)
        else:
            model.load_state_dict(checkpoint_state_dict)

        return model

    def is_loaded(self) -> bool:
        return self.isloaded

    def get_epoch(self) -> int:
        return self.epoch

    def get_total_batch(self) -> int:
        return self.total_batch

    def save(self, model, epoch, prec, total_batch):
        if not self.isloaded:
            self.logger.error(
                "You need to call load before calling save to let me know the checkpoint path. Checkpoint not saved"
            )
            return
        state_dict = {
            "epoch": epoch,
            "arch": self.arch,
            "dataset": self.dataset,
            "state_dict": model.state_dict(),
            "prec": prec,
            "total_batch": total_batch,
        }
        torch.save(state_dict, self.checkpoint_path)
        if prec > self.best_prec:
            self.best_prec = prec
            shutil.copyfile(self.checkpoint_path, self.best_path)


## 神经网络训练
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


# 计算准确度
def accuracy(y_hat, y):  # from d2l
    """计算预测正确的数量, 返回类型和y相同"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # 当y_hat每个位置表示可能性大小时
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter, device):  # from d2l
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())  # numel获取tensor中元素总个数
    return metric[0] / metric[1]


def evaluate_accuracy_loss(net, data_iter, loss, device, logger=None, epoch=-1):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(3)  # 测试损失总和、正确预测数、预测总数
    timer = Timer()
    with torch.no_grad():
        i = 0
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            metric.add(
                float(loss(y_hat, y).sum()), accuracy(y_hat, y), y.numel()
            )  # numel获取tensor中元素总个数
            if logger is not None and timer.stop() > 1.0:
                logger.trace(
                    f"[{epoch}:{i}]test loss: {(metric[0] / metric[2]):.4f}, test acc: {(metric[1] / metric[2]):.3f}"
                )
                timer.start()
            i += 1
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练一个epoch
def train_epoch_ch3(
    net,
    train_iter,
    loss,
    updater,
    device,
    logger=None,
    epoch=-1,
    tensorboard_writer=None,
    total_batch=0,
):  # from d2l
    """训练模型一个迭代周期（详见d2l第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    i = 0
    timer = Timer()
    for X, y in train_iter:
        X = X.to(device)
        y = y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        if timer.stop() > 1.0:  # 避免日志过快
            if logger is not None:
                logger.trace(
                    f"[{epoch}:{i}]train loss: {(metric[0] / metric[2]):.4f}, train acc: {(metric[1] / metric[2]):.3f}"
                )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "batch_train_loss", metric[0] / metric[2], total_batch + i
                )
                tensorboard_writer.add_scalar(
                    "batch_train_acc", metric[1] / metric[2], total_batch + i
                )
            timer.start()
        i += 1
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2], total_batch + i


# 实用animator训练全过程(from d2l)
def train_ch3(
    net, train_iter, test_iter, loss, num_epochs, updater, device, ylim=None
):  # from d2l
    """训练模型（详见d2l第3章）"""
    animator = Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        ylim=ylim,
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        train_loss, train_acc, _ = train_epoch_ch3(
            net, train_iter, loss, updater, device
        )
        test_acc = evaluate_accuracy(net, test_iter, device)
        animator.add(epoch + 1, (train_loss, train_acc) + (test_acc,))
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc


# 实用训练全过程(魔改版)
def train_ch3_plus(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    updater,
    device,
    use_animator: bool = True,
    ylim: list = None,  # 限制animator绘图的纵坐标范围，例如[0.3, 0.9]，设置为None不限制纵坐标
    tensorboard_path: str = None,  # 接受一个文件路径，设置为None不使用tensorboard
    logger: Logger = None,  # 设置为None无日志输出
    checkpoint_manager: CheckpointManager = None,  # 设置为None时不使用checkpoint
    test_freq: int = 1,  # 测试和保存checkpoint的频率
    train_batch_log: bool = False,  # 训练epoch内的每一个batch都记录log
    test_batch_log: bool = False,  # 测试epoch内的每一个batch都记录log
    batch_tensorboard: bool = False,  # 每一个训练的batch都记录到tensorboard
):  # from d2l
    """训练模型（详见d2l第3章）"""
    epoch = checkpoint_manager.get_epoch() if checkpoint_manager else 0
    epoch_limit = epoch + num_epochs
    total_batch = checkpoint_manager.get_total_batch() if checkpoint_manager else 0
    if use_animator:
        animator = Animator(
            xlabel="epoch",
            xlim=[epoch, epoch_limit],
            ylim=ylim,
            legend=["train loss", "test loss", "train acc", "test acc"],
            fmts=("r.-", "g.-", "b.-", "m.-"),
        )
    if tensorboard_path:
        writer = SummaryWriter(tensorboard_path)
    # if logger is not None:
    #     logger = Logger(output='stdout')
    if checkpoint_manager and not checkpoint_manager.is_loaded():
        raise ValueError(
            "Checkpoint manager is not loaded. Consider calling load first"
        )
    timer = Timer()
    while epoch < epoch_limit:
        # 启动计时器
        timer.start()
        # 判断本epoch是否测试
        test = epoch % test_freq == 0 or epoch == epoch_limit - 1
        # 训练
        train_loss, train_acc, total_batch = train_epoch_ch3(
            net,
            train_iter,
            loss,
            updater,
            device,
            logger if train_batch_log else None,
            epoch,
            writer if tensorboard_path and batch_tensorboard else None,
            total_batch,
        )
        # 测试
        if test:
            test_loss, test_acc = evaluate_accuracy_loss(
                net, test_iter, loss, device, logger if test_batch_log else None, epoch
            )
        # 绘制动图
        if use_animator:
            if test:
                animator.add(epoch + 1, (train_loss, test_loss, train_acc, test_acc))
            else:
                animator.add(epoch + 1, (train_loss, None, train_acc, None))
        # 绘制tensorboard
        if tensorboard_path:
            writer.add_scalar("train loss", train_loss, epoch + 1)
            writer.add_scalar("train acc", train_acc, epoch + 1)
            if test:
                writer.add_scalar("test acc", test_acc, epoch + 1)
                writer.add_scalar("test loss", test_loss, epoch + 1)
        # 结束计时器
        time_cost = timer.stop()
        # 写log
        if logger is not None:
            if test:
                logger.trace(
                    f"epoch={epoch + 1}, epoch_time={time_cost:.3f}, train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, test_loss={test_loss:.4f}, test_acc={test_acc:.3f}"
                )
            else:
                logger.trace(
                    f"epoch={epoch + 1}, epoch_time={time_cost:.3f}, train_loss={train_loss:.4f}, train_acc={train_acc:.3f}"
                )

        # 保存 checkpoint
        if test and checkpoint_manager:
            checkpoint_manager.save(net, epoch + 1, test_acc, total_batch)
        epoch += 1

    if tensorboard_path:
        writer.close()


"""Usage:
```python
import tlib
import torchvision
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dropout1, dropout2 = 0, 0

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第一个全连接层之后添加一个dropout层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第二个全连接层之后添加一个dropout层
    nn.Dropout(dropout2),
    nn.Linear(256, 10),
).to(device)
train_data = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=ToTensor(), download=True
)
test_data = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=ToTensor(), download=True
)
train_iter = torch.utils.data.DataLoader(
    train_data, batch_size=256, shuffle=True, num_workers=4
)
test_iter = torch.utils.data.DataLoader(
    test_data, batch_size=256, shuffle=False, num_workers=4
)

loss = torch.nn.CrossEntropyLoss(reduction="none").to(
    device
)  # reduction=none 设置CrossEntropyLoss 不执行平均操作，而是求和，避免loss太小导致画图不好看
num_epochs = 20
updater = torch.optim.SGD(net.parameters(), lr=0.01)
logger = tlib.Logger()
checkpoint_manager = tlib.CheckpointManager("mynet", "FashionMNIST", logger=logger)
net = checkpoint_manager.load(net, "ck.pth", "best.pth")
tlib.train_ch3_plus(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    updater,
    device,
    use_animator=True,
    tensorboard_path='./tensorboard',
    logger=None,
    checkpoint_manager=checkpoint_manager,
    test_freq=2,
)
```
"""

## 连接mysql数据库并执行SQL语句
# connect to mysql database
import mysql.connector


class database:
    # 初始化连接数据库
    def __init__(self, ip, port, user, pwd, database) -> None:
        self.conn = mysql.connector.connect(
            host=ip, port=port, user=user, password=pwd, database=database
        )
        self.cursor = self.conn.cursor()

    # 执行SQL语句并返回结果
    def exec(self, cmd: str):
        self.cursor.execute(cmd)
        result = self.cursor.fetchall()
        self.conn.commit()
        return result
