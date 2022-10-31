from dataclasses import replace
import torchvision.datasets as dset
import open_clip
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import glob
from PIL import Image
import torch.nn as nn
import copy
import wandb
from cub2100 import Cub2011
import os
import pandas as pd

print("Starting Program")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model_name = "RN50"
pretrained = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()

print(device)

dataset_path = '/home/mnpham/Downloads/CUB_200_2011'
dataset = Cub2011(root=dataset_path, train=False, download=False, transform=preprocess)

#get all classes in the dataset
classes = pd.read_csv(os.path.join(dataset_path, 'CUB_200_2011', 'classes.txt'), sep=' ',
                             names=['class_id', 'class_name'])

classes_dict = {}
classes_all = []
for row in classes.iterrows():
    classes_dict[row[1]['class_id']] = row[1]['class_name']
    classes_all.append("a photo of " + row[1].class_name.split('.')[1].replace('_', ' '))

classes_all_features = model.encode_text(open_clip.tokenize(classes_all).to(device))
classes_all_features /= classes_all_features.norm(dim=-1, keepdim=True) # 1x512

print(classes_all_features.shape)

correct = 0
for i, (image, target) in enumerate(tqdm(dataset)):
    
    image = image.unsqueeze(0).to(device)
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True) # 1x512

    #calculate similarity between image and text
    similarity = (100.0 * image_features @ classes_all_features.T).softmax(dim=-1)  # Nx2048 * 2048x1000 -> Nx1000
    values, indices = similarity[0].topk(1) # get top 5 classes

    if(indices.item() == target):
        correct += 1

    if(i == 100):
        break
    
print("Accuracy: ", correct/(i+1))