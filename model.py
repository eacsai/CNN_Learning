import torch
from torchsummary import summary
from torch.nn import functional as F
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
            nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
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


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU(),
        )

    def forward(self, X):
        return torch.cat(
            (
                self.conv1(X),
                self.conv2(X),
                self.conv3(X),
                self.conv4(X),
            ),
            dim=1
        )


class GoogleNet(nn.Module):
    def __init__(self, in_channels, classes):
        super(GoogleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Inception(192, c1=64, c2=[96, 128], c3=[16, 32], c4=32),
            Inception(256, c1=128, c2=[128, 192], c3=[32, 96], c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception(480, c1=192, c2=[96, 208], c3=[16, 48], c4=64),
            Inception(512, c1=160, c2=[112, 224], c3=[24, 64], c4=64),
            Inception(512, c1=128, c2=[128, 256], c3=[24, 64], c4=64),
            Inception(512, c1=112, c2=[144, 288], c3=[32, 64], c4=64),
            Inception(528, c1=256, c2=[160, 320], c3=[32, 128], c4=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Inception(832, c1=256, c2=[160, 320], c3=[32, 128], c4=128),
            Inception(832, c1=384, c2=[192, 384], c3=[48, 128], c4=128),
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(1024, classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1,
                               stride=strides)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y


class ResNet(nn.Module):
    def __init__(self, Residual):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False, strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 10))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le_net = LeNet().to(device)
    alex_net = AlexNet().to(device)
    google_net = GoogleNet(1, 1000).to(device)
    res_net = ResNet(Residual).to(device)

    # print(summary(le_net, (1, 28, 28)))
    # print(summary(alex_net, (1, 224, 224)))
    # print(summary(google_net, (1, 224, 224)))
    print(summary(res_net, (1, 224, 224)))

