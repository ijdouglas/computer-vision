# from __future__ import print_function
# import sys
# sys.path.append("./AttentionDeepMIL/")
import re
import pandas as pd
import os
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
import utils
import datetime

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

def save_model(model, args, loss, acc):
    if not osp.isdir(args.chkpt_dir):
        os.mkdir(args.chkpt_dir)
    subID = re.search('[0-9]{4}', str(args.train_set)).group()
    path = osp.join(args.chkpt_dir, subID + "_model.pt")
    torch.save(model.state_dict(), path)

    stats = {"loss": loss, "acc": acc}
    stats['args'] = args
    path = osp.join(args.chkpt_dir, subID + "_info.pkl")
    pickle.dump(stats, open(path, "wb"))

def setup_device(gpu_id):
    #set up GPUS
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if int(gpu_id)==-2 and os.getenv('CUDA_VISIBLE_DEVICES') is not None:
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
        print("hello")
    elif  int(gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print("set CUDA_VISIBLE_DEVICES=", gpu_id)
        print(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device %s"%device)
    
    return device




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

accuracy = lambda output,target : acc_topk(output, target)[0]    
def ACCURACY_(output, target):
    result = ACC_TOPK_(output, target)
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
        return res

def ACC_TOPK_(output, target, topk=(1,)):
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

def setup_backbone(name, pretrained=True, num_classes=2):
    if name == "vgg16":
        model = torchvision.models.vgg16(pretrained=pretrained)    
        num_features = int(model.classifier[6].in_features)
        model.classifier[6] = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet18":
        model = torchvision.models.resnet18(pretrained=pretrained)    
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)    
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    elif name == "resnet152":
        model = torchvision.models.resnet152(pretrained=pretrained)
        num_features = int(model.fc.in_features)
        model.fc = nn.Linear(num_features,num_classes)
        return model
    else:
        raise NotImplementedError("this option is not defined")

def freeze_backbone(model, name):
    print("freezing backbone")
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

    # unfreeze the final classification layer
    if name == "vgg16":
        for p in model.classifier[6].parameters():
            p.requires_grad = True
    elif name == "resnet18":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif name == "resnet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif name == "resnet152":
        for p in model.fc.parameters():
            p.requires_grad = True

    return model

def train(model, dataloader, epoch, criterion, optimizer, 
        args=None, test_loader=None, device=None, print_freq=10, 
        eval_freq=150, test_at_end=False):

    print("epoch {}".format(epoch))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train() # Set model to training mode

    losses = AverageMeter()
    accs = AverageMeter()
   
    for i, data in enumerate(dataloader):
        inputs = data["input"].to(device)
        labels = data["label"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), outputs.size(0))
        accs.update(acc.item(),outputs.size(0))

        if i % print_freq == 0 or i == len(dataloader)-1:
            temp = "current loss: %0.5f "%loss.item()
            temp += "acc %0.5f "%acc.item()
            temp += "| epoch average loss %0.5f "%losses.avg
            temp += "acc %0.5f "%accs.avg
            print(i, temp)

        # if (i % eval_freq == 0 and i > 0) or i == len(dataloader)-1:
        #     test_loss, test_acc = evaluate(test_loader, model, criterion, accuracy)
        #     save_model(model, args, test_loss, test_acc)
        #     print("\ntesting: \nloss: {}\nacc: {}\n".format(test_loss, test_acc))

    if args.test_at_end:
        test_loss, test_acc = evaluate(test_loader, model, criterion, accuracy)
        print("\ntesting: \nloss: {}\nacc: {}\n".format(test_loss, test_acc))
        if args.chkpt_dir is not None:
            print("saving model...")
            save_model(model, args, test_loss, test_acc)
# IJD: edit evaluate() to reflect the version in Andrei's test.py script
# and further edit it to write out the results data frame that is created.
# Add an argument 'output' to evaluate() to then store results in folder: output
def evaluate(dataloader, model, criterion, accuracy, output, device=None, num_classes = 2):        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    model.eval()
    if num_classes == 2:
        # Assume we're modeling joint attention v not joint attention
        results = pd.DataFrame(columns=['img','notJA_prob', 'JA_prob'])
        #results = pd.DataFrame(columns=['img', "model_classif", "JA_prob", "notJA_prob",  "prob", "correct", "correct_label", "predict_label"])
    else:
        # We are doing multiclass classification
        results = pd.DataFrame(columns = ['img'] + ['class_' + str(n) for n in range(num_classes)]) 

    eval_fun = nn.Softmax()
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            inputs = data["input"].to(device)
            labels = data["label"].to(device)
            img_path = data["img_path"]
            outputs = model(inputs)
            probs = eval_fun(outputs)
            #print(' Probs: ')
            #print(probs)
            #print(' Img path: ')
            #print(img_path)
            loss = criterion(outputs, labels)
            #acc = accuracy(outputs, labels)
            acc, preds, correct = accuracy(outputs, labels)
            #print(' Accuracy: ')
            #print(acc)
            #print(' Preds: ')
            #print(preds)
            #print(' Correct: ')
            #print(correct)
            res_ = pd.DataFrame({"img": img_path})
            probs_ = probs.cpu() # copy the tensor to the CPU from the device (from the gpu)
            if num_classes == 2:
                # Assume it's modeling joint attention v not joint attention
                prob_res_ = pd.DataFrame(probs_.numpy(), columns=['notJA_prob', 'JA_prob'])
            else:
                prob_res_ = pd.DataFrame(probs_.numpy(), columns=['class_' + str(n) for n in range(num_classes)])
            results = results.append(pd.concat([res_, prob_res_], axis=1))
            #results = results.append({
            #    "img": img_path[0],
            #    #"model_classif": "JA" if int(preds[0].item()) == 1 else "not_JA",
            #    #"notJA_prob": float(probs[0][0]),
            #    "JA_prob": float(probs[0][1]),
            #    "notJA_prob": float(probs[0][0]),
            #    #"prob": float(probs[0][preds[0]]),
            #    "correct": bool(correct[0]),
            #    "correct_label": int(labels[0])#,
            #    #"predict_label": int(preds[0].item())
            #}, ignore_index=True)
            losses.update(loss.item(), outputs.size(0))
            accs.update(acc.item(),outputs.size(0))
            
        # Save results to csv
        results.to_csv(output, index=False)

    # print("eval loss %0.5f acc %0.5f "%(losses.avg,accs.avg))    
    return float(losses.avg), float(accs.avg)


def main(args):
    args.seed = utils.setup_seed(args.seed)
    args.batch = int(args.batch)
    utils.make_deterministic(args.seed)

    print('Load Train and Test Set')
    train_loader, test_loader, train_set, test_set = setup_dataloader(args)
    print("num training frames: ", len(train_loader) * args.batch)
    print("batch size: ", args.batch)

    device = setup_device(args.gpu)
    print("Device: ", torch.cuda.get_device_name(device))

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        print('\nGPU is ON!')

    print('Init Model')
    model = setup_backbone(name=args.model, num_classes=args.num_classes, pretrained = True)
    if args.freeze_backbone:
        model = freeze_backbone(model, args.model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.loss_balanced:  
        print("using balanced loss")
        #if this optin is true, weight the loss inversely proportional to class frequency
        weight = torch.FloatTensor(train_set.inverse_label_freq)
        criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device)

    print('Start Training')
    ct = datetime.datetime.now()
    print("current time: ", ct)
    print('Target Obj: {}'.format(args.target_number))
    print("Epochs: {}".format(args.epochs))
    print("Weight Decay: {}".format(args.reg))
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, epoch, criterion, optimizer, device=device, args=args, test_loader=test_loader, test_at_end=args.test_at_end)
    print('Start Testing')
    # IJD: Create additional path to the folder to store the results
    subID = re.search('[0-9]{4}', str(args.train_set)).group()
    #output_ = args.train_set.replace('data', 'results').replace('train.csv', 'results.csv').replace('train/', 'results/')
    output_ = os.path.join(args.results_dir, subID + '_results.csv')
    test_loss, test_acc = evaluate(test_loader, model, criterion, accuracy=ACCURACY_, output=output_, device=device, num_classes=args.num_classes)
    print("\nSubject {} testing results: \nloss: {}\nacc: {}\n".format(subID, test_loss, test_acc))
    if args.chkpt_dir is not None:
        save_model(model, args, test_loss, test_acc)

    return model, test_loss, test_acc



if __name__ == "__main__":
    args = setup_args()
    if args.config is not None:
        with open(args.config, "r") as input:
            cfg = json.load(input)
            args = load_config(args, cfg)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model, loss, acc = main(args)
