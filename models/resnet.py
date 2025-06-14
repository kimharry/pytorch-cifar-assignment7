'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=None, weight_init=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        if weight_init == 'gaussian':
            self.weight_init = nn.init.normal_
        elif weight_init == 'xavier':
            self.weight_init = nn.init.xavier_normal_
        elif weight_init == 'kaiming':
            self.weight_init = nn.init.kaiming_normal_
        else:
            raise ValueError('Invalid weight initialization method')

        self.weight_init(self.conv1.weight)
        self.weight_init(self.conv2.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        for m in self.shortcut:
            if isinstance(m, nn.Conv2d):
                self.weight_init(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation=None, weight_init=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        if weight_init == 'gaussian':
            self.weight_init = nn.init.normal_
        elif weight_init == 'xavier':
            self.weight_init = nn.init.xavier_normal_
        elif weight_init == 'kaiming':
            self.weight_init = nn.init.kaiming_normal_
        else:
            raise ValueError('Invalid weight initialization method')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, activation=activation, weight_init=weight_init)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, activation=activation, weight_init=weight_init)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, activation=activation, weight_init=weight_init)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, activation=activation, weight_init=weight_init)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.weight_init(self.conv1.weight)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        self.weight_init(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, activation=None, weight_init=None):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation, weight_init=weight_init))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(activation=None, weight_init=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], activation=activation, weight_init=weight_init)


def ResNet34(activation=None, weight_init=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], activation=activation, weight_init=weight_init)


def test():
    net = ResNet18(activation='relu', weight_init='gaussian')
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
