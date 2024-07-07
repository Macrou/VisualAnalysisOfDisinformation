import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils
import torchvision.transforms as transforms
from torchvision import datasets 
import numpy as np
import matplotlib.pyplot as plt

import argparse
from models import *
from utils import progress_bar, get_mean_and_std
from dataloaders.fakeddit_data_loader import *
from torchvision.models import resnet34,ResNet34_Weights,resnet50,ResNet50_Weights

parser = argparse.ArgumentParser(description='PyTorch Disinformation Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ep',default=15,type=int,help='maximum epoch')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--data','-d', choices=['Fakeddit', 'CIFAKE'],default='Fakeddit', type= str, help='data set')
parser.add_argument('--test', '-t', action='store_true',
                    help='test the model without training')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4693, 0.4366, 0.4082), (0.2375, 0.2286, 0.2279)),
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4707, 0.4399, 0.4118), (0.2375, 0.2286, 0.2279)),
])

if args.data == 'Fakeddit':  
    trainset = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv",transform=transform_train)
    testset =  Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_test_public.tsv",transform=transform_test)
elif args.data == 'CIFAKE':
    trainset = datasets.ImageFolder(root='dataset/train',transform=transform_train)
    testset = datasets.ImageFolder(root='dataset/test',transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset,batch_size=128, shuffle=True, num_workers=4
)

# testset = Fakeddit(annotations_file="./dataset/multimodal_only_samples/multimodal_validate.tsv",
#                    transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset,batch_size=100, shuffle=False, num_workers=4
)

classes = ('false','real')

trainlossplt = np.array([])
testlossplt = np.array([])
trainaccuracy = np.array([])
testaccuracy = np.array([])

# Model
print('==> Building model..')
net = resnet34(weights=ResNet34_Weights.DEFAULT)
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
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-8)

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
        # print(outputs_1.shape)
        # plt.imshow(outputs_1[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        print(batch_idx)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    global trainlossplt,trainaccuracy
    trainlossplt = np.concatenate((trainlossplt,np.array([train_loss])))
    trainaccuracy = np.concatenate((trainaccuracy,np.array([100.*correct/total])))

def test(epoch):
    global best_acc,testlossplt,testaccuracy
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    testlossplt = np.concatenate((testlossplt,np.array([test_loss])))
    testaccuracy = np.concatenate((testaccuracy,np.array([100.*correct/total])))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('results/checkpoint'):
            os.mkdir('results/checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, './results/checkpoint/ckpt.pth')

        best_acc = acc

def plot_figures():
    epochs = range(start_epoch, start_epoch+args.ep)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, trainlossplt, 'bo-', label='Training Loss')
    plt.plot(epochs, testlossplt, 'ro-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname='results/plots/loss.png',format='png')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, trainaccuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, testaccuracy, 'ro-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname='results/plots/trainingAccuracy.png',format='png')

def print_mean_and_div():
    print('==> Calculating mean for training')
    trainmean,trainstd = get_mean_and_std(trainset)
    print('==> Calculating mean for test')
    testmean,trainstd =    get_mean_and_std(testset)
    print(f"the mean and deviation for training are {trainmean} {trainstd} and for test are {testmean} and {trainstd}")

if __name__ == "__main__":
    if args.train :
        test(start_epoch)
    else:
        for epoch in range(start_epoch, start_epoch + args.ep):
            train(epoch)
            test(epoch)
            plot_figures()

