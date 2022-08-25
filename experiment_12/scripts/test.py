import os
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
import utils
import pandas as pd
from tqdm import tqdm


import argparse
import os.path as osp
import torch
from torch import nn
import torch.utils.data as data_utils
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datasets.ImagePandasDataset import ImagePandasDataset 
from torch.utils.data import Dataset, DataLoader
import torchvision


from args import setup_args, load_config
from dataloader import Egocentric, setup_dataloader

import train as t

# accuracy = lambda output,target : acc_topk(output, target)[0]    

def accuracy(output, target):
    result = acc_topk(output, target)
    acc = result[0][0]
    preds = result[1]
    correct = result[2]
    return acc, preds, correct

#taken from pytorch imagenet example 
def acc_topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res, pred, correct


def setup_dataloader(args):
    test = Egocentric(csv=args.test_set,
                target_number=args.target_number,
                root_dir=args.root_dir,
                transform=transforms.Compose([
                    transforms.Resize((256, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))]),
                seed=args.seed,
                train=True)

    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    test_loader = data_utils.DataLoader(test, batch_size=args.batch,
                                        shuffle=True,
                                        **loader_kwargs)

    return test_loader, test


def main(args):
    args.seed = utils.setup_seed(args.seed)
    utils.make_deterministic(args.seed)

    print('Load Train and Test Set')
    test_loader, test_set = setup_dataloader(args)
    print("num testing frames: ", len(test_loader))

    device = t.setup_device(args.gpu)
    print("Device: ", torch.cuda.get_device_name(0))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Init Model')
    model = t.setup_backbone(args.model)

    model.load_state_dict(torch.load(args.saved_model))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_loss, test_acc, results = evaluate(test_loader, model, criterion, accuracy=accuracy, device=device)
    return test_loss, test_acc, results

def evaluate(dataloader, model, criterion, accuracy, device=None):        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = pd.DataFrame(columns=['img', "model_classif", "learn_prob", "notlearn_prob",  "prob", "correct", "correct_label", "predict_label"])

    softmax = nn.Softmax()
    
    losses = t.AverageMeter()
    accs = t.AverageMeter()
    with torch.no_grad():
        for data in tqdm(dataloader):
            inputs = data["input"].to(device)
            labels = data["label"].to(device)
            img_path = data["img_path"]
            outputs = model(inputs)
            probs = softmax(outputs)
            loss = criterion(outputs, labels)
            acc, preds, correct = accuracy(outputs, labels)
            results = results.append({
                "img": img_path[0],
                "model_classif": "learned" if int(preds[0]) == 1 else "not_learned",
                "notlearn_prob": float(probs[0][0]),
                "learn_prob": float(probs[0][1]),
                "prob": float(probs[0][preds[0]]),
                "correct": bool(correct[0]),
                "correct_label": int(labels[0]),
                "predict_label": int(preds[0])
            }, ignore_index=True)
            losses.update(loss.item(), outputs.size(0))
            accs.update(acc.item(), outputs.size(0))

    # print("eval loss %0.5f acc %0.5f "%(losses.avg,accs.avg))    
    return float(losses.avg), float(accs.avg), results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()