import torch
from torchsummary import summary
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer = nn.Sequential(
            # 这里使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Vgg(nn.Module):
    def __init__(self, conv_arch):
        super(Vgg, self).__init__()
        conv_blks = []
        in_channels = 1

        # 定义一个vgg块
        def vgg_block(vgg_num_convs, vgg_in_channels, vgg_out_channels):
            layers = []
            for _ in range(vgg_num_convs):
                layers.append(nn.Conv2d(vgg_in_channels, vgg_out_channels,
                                        kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                vgg_in_channels = vgg_out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.layers = nn.Sequential(
            *conv_blks,
            nn.Flatten(),
            # 全连接层部分
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le_net = LeNet().to(device)
    alex_net = AlexNet().to(device)

    # print(summary(le_net, (1, 28, 28)))
    # print(summary(alex_net, (1, 224, 224)))
