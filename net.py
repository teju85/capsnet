#!/usr/bin/env python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import time
from torch.optim import Adam, SGD
import math

def get_datasets(root):
    """
    Helper function to download, prepare and return the MNIST datasets for
    training/testing, respectively.
    Args:
    . root - folder where to download and prepare the dataset
    Output: training and testing sets, respectively
    """
    trainset = MNIST(root, train=True, download=True, transform=ToTensor())
    testset = MNIST(root, train=False, download=True, transform=ToTensor())
    return trainset, testset

def get_loaders(trainset, testset, batch_size, test_batch_size, shuffle):
    """
    Prepare DataLoader wrappers for training and testing set, respectively.
    Args:
    . trainset - training set
    . testset - testing set
    . batch_size - batch size during training
    . test_batch_size - batch size during testing
    . shuffle - whether to shuffle inputs
    Output: dataloaders for training set and testing set, respectively
    """
    pin_memory = True if torch.cuda.is_available() else False
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                              pin_memory=pin_memory)
    test_loader = DataLoader(testset, batch_size=test_batch_size,
                             shuffle=shuffle, pin_memory=pin_memory)
    return train_loader, test_loader

class Reconstructor(nn.Module):
    def __init__(self, nCaps, capsDim):
        super(Reconstructor, self).__init__()
        self.nCaps = nCaps
        self.capsDim = capsDim
        self.fc1 = nn.Linear(nCaps*capsDim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, x, labels):
        idx = Variable(torch.zeros(x.size(0), self.nCaps), requires_grad=False)
        if x.is_cuda:
            idx = idx.cuda()
        idx.scatter_(1, labels.view(-1, 1), 1)  # one-hot vector!
        idx = idx.unsqueeze(dim=-1)
        activities = x * idx
        activities = activities.view(x.size(0), self.nCaps*self.capsDim)
        x = F.relu(self.fc1(activities))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.view(x.size(0), 1, 28, 28)
        return x

def squash(x, dim=-1):
    norm2 = x.pow(2).sum(dim=dim, keepdim=True)
    scale = norm2 / (1 + norm2) / norm2.sqrt()
    x = scale * x
    return x

class ConvCapsule(nn.Module):
    def __init__(self, inC, outC, capsDim, stride, kernel):
        super(ConvCapsule, self).__init__()
        self.outC = outC
        self.capsDim = capsDim
        arr = []
        self.c1 = nn.Conv2d(inC, outC*capsDim, kernel_size=kernel,stride=stride)

    def forward(self, x):
        out = self.c1(x)
        N, _, H, W = out.size()
        out = out.view(N, self.outC, self.capsDim, H, W)
        out = out.permute([0, 1, 3, 4, 2])
        a, b, c, d, e = out.size()
        out = out.contiguous()
        out = out.view(a, b*c*d, e)
        out = squash(out)
        return out

class Capsule(nn.Module):
    def __init__(self, nOutCaps, outCapsDim, nInCaps, inCapsDim, nRouting):
        super(Capsule, self).__init__()
        self.nOutCaps = nOutCaps
        self.outCapsDim = outCapsDim
        self.nInCaps = nInCaps
        self.inCapsDim = inCapsDim
        self.r = nRouting
        self.W = nn.Parameter(torch.randn(nInCaps, inCapsDim, nOutCaps * outCapsDim))
        stdv = 1. / math.sqrt(nInCaps)
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, u):
        b = Variable(torch.zeros(u.size(0), self.nInCaps, self.nOutCaps))
        if torch.cuda.is_available():
            b = b.cuda()
        u1 = u.unsqueeze(dim=2)
        uhat = u1.matmul(self.W)
        uhat = uhat.view(uhat.size(0), self.nInCaps, self.nOutCaps, self.outCapsDim)
        for i in range(self.r):
            c = F.softmax(b, dim=-1)
            c = c.unsqueeze(3)
            s = torch.sum(c * uhat, dim=1)
            v = squash(s)
            if i != self.r - 1:
                v1 = v.unsqueeze(1)
                a = torch.sum(uhat * v1, dim=-1)
                b = b + a
        return v

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.c1 = nn.Conv2d(1, 256, kernel_size=9)
        self.convcaps = ConvCapsule(inC=256, outC=32, capsDim=8, stride=2,
                                    kernel=9)
        self.caps = Capsule(nOutCaps=10, outCapsDim=16, nInCaps=32*6*6,
                            inCapsDim=8, nRouting=3)
        self.decoder = Reconstructor(nCaps=10, capsDim=16)

    def forward(self, x, labels=None):
        x = self.c1(x)
        x = F.relu(x)
        x = self.convcaps(x)
        x = self.caps(x)
        pred = torch.sqrt(torch.sum(x.pow(2), dim=-1))
        if labels is not None:
            recon = self.decoder(x, labels)
        else:
            recon = None
        return pred, recon, x

class MarginLoss(nn.Module):
    def __init__(self, mplus, _lambda, mminus, recon_weight):
        super(MarginLoss, self).__init__()
        self.mplus = mplus
        self._lambda = _lambda
        self.mminus = mminus
        self.recon_weight = recon_weight

    def forward(self, output, data, label):
        pred, recon, x = output
        idx = torch.zeros(pred.size())
        if pred.is_cuda:
            idx = idx.cuda()
        idx = idx.scatter_(1, label.data.view(-1, 1), 1) # one-hot!
        idx = Variable(idx)
        idx = idx.float()
        loss_plus = F.relu(self.mplus - pred).pow(2)
        loss_minus = F.relu(pred - self.mminus).pow(2)
        loss = (idx * loss_plus) + (self._lambda * (1. - idx) * loss_minus)
        lval = loss.sum()
        if recon is not None:
            return lval + self.recon_weight * F.mse_loss(recon, data)
        return lval


def train(epoch_id, model, loader, loss, optimizer, recon):
    start = time.time()
    loss_val = 0.0
    accuracy = 0.0
    model.train()
    for idx, (data, label) in enumerate(loader):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()
        if recon:
            output = model(data, label)
        else:
            output = model(data)
        lval = loss(output, data, label)
        lval.backward()
        optimizer.step()
        loss_val += lval.data[0]
        _, pred = output[0].data.max(dim=-1)  # argmax
        accuracy += pred.eq(label.data.view_as(pred)).sum()
    loss_val /= len(loader.dataset)
    accuracy /= len(loader.dataset)
    total = time.time() - start
    print("Train epoch:%d time(s):%.3f loss=%.8f accuracy:%.4f" % \
          (epoch_id, total, loss_val, accuracy))

def test(epoch_id, model, loader, loss):
    start = time.time()
    model.eval()
    loss_val = 0.0
    accuracy = 0.0
    for idx, (data, label) in enumerate(loader):
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        output = model(data)
        loss_val += loss(output, data, label).data[0]
        _, pred = output[0].data.max(1)  # argmax
        accuracy += pred.eq(label.data.view_as(pred)).sum()
    loss_val /= len(loader.dataset)
    accuracy /= len(loader.dataset)
    total = time.time() - start
    print("Test epoch:%d time(s):%.3f loss=%.8f accuracy:%.4f" % \
          (epoch_id, total, loss_val, accuracy))

if __name__ == "__main__":
    import argparse
    print("Parsing args...")
    parser = argparse.ArgumentParser(description="Capsnet Benchmarking")
    parser.add_argument("-adam", default=False, action="store_true",
                        help="Use ADAM as the optimizer (Default SGD)")
    parser.add_argument("-batch-size", type=int, default=128,
                        help="Input batch size for training")
    parser.add_argument("-epoch", type=int, default=10, help="Training epochs")
    parser.add_argument("-lr", type=float, default=0.1, help="Learning Rate")
    parser.add_argument("-mom", type=float, default=0.9,
                        help="Momentum (SGD only)")
    parser.add_argument("-recon", default=False, action="store_true",
                        help="Enable reconstruction loss")
    parser.add_argument("-root", type=str, default="mnist",
                        help="Directory where to download the mnist dataset")
    parser.add_argument("-seed", type=int, default=12345,
                        help="Random seed for number generation")
    parser.add_argument("-shuffle", default=False, action="store_true",
                        help="To shuffle inputs during training/testing or not")
    parser.add_argument("-test-batch-size", type=int, default=256,
                        help="Input batch size for testing")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print("Loading datasets...")
    trainset, testset = get_datasets(args.root)
    train_loader, test_loader = get_loaders(trainset, testset, args.batch_size,
                                            args.test_batch_size, args.shuffle)
    print("Preparing model/loss-function/optimizer...")
    model = CapsuleNetwork()
    if torch.cuda.is_available():
        model.cuda()
    loss = MarginLoss(0.9, 0.5, 0.1, 0.0005)
    if args.adam:
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    print("Training loop...")
    for idx in range(0, args.epoch):
        train(idx, model, train_loader, loss, optimizer, args.recon)
        test(idx, model, test_loader, loss)
