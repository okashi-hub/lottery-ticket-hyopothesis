
import argparse
import statistics

import copy
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from  models import lanet
import utils

#endless loader
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


#logger
from datetime import datetime

def logging(fname, text):
    f = open(fname,"a")
    out_text = "[%s]\t%s" %(str(datetime.now()),text)
    f.write(out_text+"\n")
    # print (out_text)
    f.close()


def main(args, ITE=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resample = True if args.prune_type=="resample" else False
    reinit = True if args.prune_type=="reinit" else False

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0,drop_last=False)
    train_loader = cycle(train_loader)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0,drop_last=True)

    #モデル作成＆初期値保存
    models = []
    initials = []
    optimizers = []
    for i in range(3):
        models.append(lanet.LeNet(mask=True).to(device))
        initial_state_dict = copy.deepcopy(models[i].state_dict())
        initials.append(initial_state_dict)
        torch.save({"state_dict": initial_state_dict}, f"saves/{i}_model/initial_state_dict_{args.prune_type}.pth.tar")
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=args.lr, weight_decay=1e-4))
    
    # print("Model's initial state_dict:")
    # for model in models:
    #     for key, param in model.state_dict().items():
    #         print(key, '\t', param.size())


    start_iteration = 0

    #未改良　再開不可
    if args.resume:
        for i, model in enumerate(models):
            checkpoint = torch.load(f"saves/{i}_model/model_{args.prune_type}.pth.tar")
            start_iteration = checkpoint["iter"]
            model.load_state_dict(checkpoint["state_dict"])
            initial_state_dict = torch.load(f"saves/{i}_model/initial_state_dict_{args.prune_type}.pth.tar")["state_dict"]
            optimizers[i] = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ITERATION = 25
    for _ite in range(start_iteration, ITERATION):
        # Pruning
        for i, model in enumerate(models):
            if not _ite == 0:
                model.prune_by_percentile(resample=resample, reinit=reinit)
                if not reinit:
                    initial_state_dict = initials[i]
                    utils.original_initialization(model, initial_state_dict)
                optimizers[i] = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"--- Pruning Level [{ITE}:{_ite}/15]: {1.0*(0.8**(_ite))*100:.1f}---")
        utils.print_nonzeros(model)                 #刈り込まれた量を表示する．本当は全体で平均を取るべき

        # Training & Test
        pbar = tqdm(range(args.end_iter))
        for iter_ in pbar:
            if iter_ % args.valid_freq == 0:
                accuracies = test(models, test_loader)
                # ens_acc = test_ensemble(models, test_loader)
                logging(f"log/{ITE}_reinit/train_{_ite}_{args.prune_type}.log",\
                         "[%06d]\tmodel0's Acc = %04f, model1's Acc = %04f , model2's Acc = %04f" % (iter_, float(accuracies[0]), float(accuracies[1]), float(accuracies[2])))
            losses = train(models, train_loader, optimizers)
            # print("losses type:",type(losses), "loss type:", type(losses[0]), "accuracies type:", type(accuracies))
            if iter_ % args.valid_freq == 0:
                # print(f'Train Epoch: {iter_}/{args.end_iter} Loss: {statistics.mean(losses):.6f} Accuracy: {statistics.mean(accuracies):.2f}%')
                pbar.set_description(
                    f'Train Epoch: {iter_}/{args.end_iter} Loss: {statistics.mean(losses):.6f} Accuracy: {statistics.mean(accuracies):.2f}%')

        for i, model in enumerate(models):    
            torch.save({"state_dict": model.state_dict(), "iter": _ite}, f"saves/{i}_model/{_ite}_model_{args.prune_type}.pth.tar")


def train(models, train_loader, optimizers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #勾配の初期化
    for optimizer in optimizers:
        optimizer.zero_grad()

    outputs = []
    train_losses = []

    imgs, targets = next(train_loader)
    imgs, targets = imgs.to(device), targets.to(device)
    for i, model in enumerate(models):
        model.train()
        output = F.log_softmax(model(imgs), dim=1)
        train_loss = F.nll_loss(output, targets)
        train_loss.backward()
        
        #保存
        outputs.append(output)
        train_losses.append(train_loss)

        # zero-out all the gradients corresponding to the pruned connections
        for name, p in model.named_parameters():
            if 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            grad_tensor = p.grad.data.cpu().numpy()
            grad_tensor = np.where(tensor == 0, 0, grad_tensor)
            p.grad.data = torch.from_numpy(grad_tensor).to(device)
        optimizers[i].step()
    
    for i, train_loss in enumerate(train_losses):
        train_losses[i] = train_loss.item()
        # print(type(train_loss))

    return train_losses


def test(models, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracies = []
    for i, model in enumerate(models):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = F.log_softmax(model(data), dim=1)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
        accuracies.append(accuracy)
    return accuracies

def test_ensemble(models, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, model in enumerate(models):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default= 1.2e-3, type=float)
    parser.add_argument("--batch_size", default=60, type=int)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=4000, type=int)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--valid_freq", default=100, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prune_type", default="normal", type=str, help="normal | resample | reinit")


    args = parser.parse_args()

    main(args, ITE=0)
    # for i in range(0, 5):
    #     main(args, ITE=i)
