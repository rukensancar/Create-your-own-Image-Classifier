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

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str,
                        help='path to folder of images')
parser.add_argument('--arch', type=str, default='alexnet', 
                        help='CNN model architecture')
parser.add_argument('--gpu', action='store', default='gpu', 
                        help='Use GPU')
parser.add_argument('--topk', type=int, 
                        help='Top K')
parser.add_argument('--checkpoint', action='store', default='checkpoint.pth', 
                        help='Checkpoint')
args = parser.parse_args ()
path = args.dir

# TODO: Write a function that loads a checkpoint and rebuilds the model
def rebuild_model(path):
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    model,_,_ = nn_measure(structure , 0.5, hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    proc_pil = Image.open(image)
    
    preprocess_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = preprocess_image(proc_pil)
    
    return img
    # TODO: Process a PIL image for use in a PyTorch model
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

    # TODO: Implement the code to predict the class from an image file
    
def check_sanity():
    plt.rcParams["figure.figsize"] = (10,5)
    plt.subplot(211)
    
    index = 1
    path = test_dir + '/19/image_06196.jpg'

    probabilities = predict(path, model)
    image = process_image(path)
    probabilities = probabilities
    

    axs = imshow(image, ax = plt)
    axs.axis('off')
    axs.title(cat_to_name[str(index)])
    axs.show()
    
    
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(probabilities[1][0])]
    
    
    N=float(len(b))
    fig,ax = plt.subplots(figsize=(8,3))
    width = 0.8
    tickLocations = np.arange(N)
    ax.bar(tickLocations, a, width, linewidth=4.0, align = 'center')
    ax.set_xticks(ticks = tickLocations)
    ax.set_xticklabels(b)
    ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
    ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
    ax.set_ylim((0,1))
    ax.yaxis.grid(True)
    
    plt.show()