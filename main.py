'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

import os
import argparse

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--num_epochs', '-n', default=100, type=int, help='number of epochs to train')

# Activation Function - relu, leaky_relu, tanh
parser.add_argument('--activation', '-a', default='relu', type=str, help='activation function')

# Data Preprocessing - mean image, per-channel mean, per-channel mean + per-channel std
parser.add_argument('--data_preprocessing', '-d', default='mean_img', type=str, help='data preprocessing') 

# Weight Initialization - Gaussian, Xavier, Kaiming
parser.add_argument('--weight_init', '-w', default='gaussian', type=str, help='weight initialization')

parser.add_argument('--lr', '-l', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
mean_img = 0.4734
mean_per_channel_img = torch.tensor([0.4914, 0.4822, 0.4465])
std_per_channel_img = torch.tensor([0.2470, 0.2435, 0.2616])

if args.data_preprocessing == 'mean_img':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_img, 1),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_img, 1),
    ])

elif args.data_preprocessing == 'per_channel_mean':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel_img, torch.tensor([1.0, 1.0, 1.0])),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_img, 1),
    ])

elif args.data_preprocessing == 'per_channel_mean_std':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel_img, std_per_channel_img),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_per_channel_img, std_per_channel_img),
    ])

else:
    raise ValueError('Invalid data preprocessing method')

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=6)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=6)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18(activation=args.activation, weight_init=args.weight_init)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

writer = SummaryWriter('./logs/'+args.activation+'_'+args.data_preprocessing+'_'+args.weight_init+'_'+str(args.num_epochs))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    writer.add_scalar('train/loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('train/accuracy', 100*correct/total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    writer.add_scalar('test/loss', test_loss/(batch_idx+1), epoch)
    writer.add_scalar('test/accuracy', 100*correct/total, epoch)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.activation+'_'+args.data_preprocessing+'_'+args.weight_init+'.pth')
        best_acc = acc


for epoch in range(args.num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)

writer.close()
