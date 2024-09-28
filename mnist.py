from modules.transform import Transform
from modules import layers
from modules.Base import BaseModel, BaseLayer
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F


RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
MAGENTA = "\033[95m"
SEQ_LEN = 100000
trans = Transform(SEQ_LEN)


def fit(model: nn.modules.Module, optim, lossfunc, trainloader: DataLoader):
    model.train()
    totalloss = 0
    for data, target in trainloader:
        data, target = data.cuda(), target.cuda()
        data /= data.abs().max()
        optim.zero_grad()
        output = model(data)
        loss = lossfunc(output, target)
        loss.backward()
        optim.step()
        for module in model.modules():
            if isinstance(module, layers.StreamLinear):
                module.weight.data.clip_(-1, 1)

        with torch.no_grad():
            totalloss += loss.item() * data.size(0)
    return totalloss / len(trainloader.sampler)


def evaluate(
    model: nn.modules.Module, val_loader: DataLoader, lossfunc: nn.CrossEntropyLoss
):
    model.eval()
    loss = 0
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs /= inputs.abs().max()
            outputs = model(inputs)
            loss += lossfunc(outputs, labels).item() * inputs.size(0)
            _, predicted_top1 = torch.max(outputs, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()
            total += labels.size(0)
        loss /= len(val_loader.sampler)
        acc_top1 = correct_top1 / total
    return loss, acc_top1


def Sevaluate(model: BaseModel, val_loader: DataLoader):
    model.eval().cpu()
    model.module.generate_Sparams()
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = trans.f2s(inputs / inputs.abs().max())
            outputs = trans.s2f(model.module.Sforward(inputs))
            _, predicted_top1 = torch.max(outputs, 1)
            correct_top1 += (predicted_top1 == labels).sum().item()
            total += labels.size(0)
        acc_top1 = correct_top1 / total
    model.cuda()
    return acc_top1


class DNN(BaseModel):
    def __init__(self, seq_len):
        super().__init__(seq_len)
        self.fn1 = layers.StreamLinear(28 * 28, 128, seq_len)
        self.fn2 = layers.StreamLinear(128, 10, seq_len)
        self.ac1 = layers.BTanh(seq_len, 28 * 28)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28 * 28)
        x = self.fn1(x)
        x = self.ac1(x)
        x = self.drop(x)
        x = self.fn2(x)
        return x

    def Sforward(self, stream: torch.Tensor):
        x = stream.view(-1, 28 * 28, self.seq_len)
        x = self.fn1.Sforward(x)
        x = self.ac1.Sforward(x, 2300)
        x = self.fn2.Sforward(x)
        return x


path = "/data/home/wwang/projs/binary/data"


def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=path, train=True, download=False, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=path, train=False, download=False, transform=transform
    )

    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_samples))
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler, num_workers=16
    )
    val_loader = DataLoader(
        train_dataset, batch_size=128, sampler=val_sampler, num_workers=16
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=16
    )
    return train_loader, val_loader, test_loader


device = torch.device("cuda")
model = DNN(SEQ_LEN)
lossfunc = nn.CrossEntropyLoss().to(device)
train_loader, val_loader, test_loader = load_data()

lr = 0.1
counter = 0
min_val_loss = np.inf

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
else:
    model.cuda()
    print("Using single GPU")

min_val_loss, val_acc1 = evaluate(model, test_loader, lossfunc)
print(f"init: val_loss: {min_val_loss}, top1_acc:{val_acc1}")

for epoch in range(100):
    if counter / 5 == 1:
        counter = 0
        lr *= 0.5
        print(GREEN + f"lr reduced to {lr}" + RESET)
    optimi = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss = fit(model, optimi, lossfunc, train_loader)
    val_loss, val_acc1 = evaluate(model, val_loader, lossfunc)
    print(f"epoch: {epoch+1}, loss: {loss}, val_loss: {val_loss}, acc:{val_acc1}")
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
        test_loss, test_acc1 = evaluate(model, test_loader, lossfunc)
        print(MAGENTA + f"test perf: {test_acc1}" + RESET)
        torch.save(model.state_dict(), "bestmodel.pth")
    else:
        counter += 1

model.load_state_dict(torch.load("bestmodel.pth"))
S_test_acc1 = Sevaluate(model, test_loader)
print(MAGENTA + f"SC test perf: {S_test_acc1}" + RESET)
