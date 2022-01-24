import torch
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch import nn
from torch import optim
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse


arch = {"vgg16":25088,
        "densenet121" : 1024,
        "alexnet" : 9216 }

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='flowers/', 
                        help='path to folder of images')
parser.add_argument('--arch', type=str, default='alexnet', 
                        help='CNN model architecture')
parser.add_argument('--gpu', action='store', default='gpu', 
                        help='Use GPU')
parser.add_argument('--checkpoint', action='store', default='checkpoint.pth', 
                        help='Checkpoint')
args = parser.parse_args ()
data_dir = args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


if data_dir:
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    # TODO: Load the datasets with ImageFolder

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)


    # TODO: Using the image datasets and the trainforms, define the dataloaders

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
arch = {"vgg16":25088,
        "densenet121" : 1024,
        "alexnet" : 9216 }

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nn_measure(structure = 'alexnet', dropout = 0.5, hidden_layer1 = 120, lr = 0.001):
    
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)        
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.".format(structure))


    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.to(device);
    
    return model, optimizer, criterion

model, optimizer, criterion = nn_measure()

epochs = 12
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
   
    for inputs, labels in train_dataloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(valid_dataloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}.. "
                   "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_dataloader)*100))
            
                
            running_loss = 0
            #model.train()

# TODO: Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx
model.cpu
torch.save({'structure' :'alexnet',
            'hidden_layer1':500,
            'state_dict':model.state_dict(),
            'class_to_idx':model.class_to_idx},
            'checkpoint.pth')