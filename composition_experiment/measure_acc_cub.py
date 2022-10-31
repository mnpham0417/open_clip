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
from torch.autograd import Variable

print("Starting Program")

def accuracy(model_name, pretrained):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model_name = "RN50"
        # pretrained = "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()

        dataset_path = '/scratch/mp5847/CUB_200_2011/'
        dataset = Cub2011(root=dataset_path, train=False, download=False, transform=preprocess)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=5)

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
        
        correct = 0
        total = 0
        for i, (image, target) in enumerate(tqdm(dataloader)):
            
            image = image.to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True) # 1x512

            #calculate similarity between image and text
            similarity = (100.0 * image_features @ classes_all_features.T).softmax(dim=-1)  # Nx2048 * 2048x1000 -> Nx1000
            
            similarity = similarity.detach().cpu() #convert to numpy array

            #get top 1 predictions
            values, indices = similarity.topk(1, dim=-1) # Nx1
            indices = indices.squeeze(1) # Nx1 -> Nx

            correct += (indices == target).sum().item()
            total += image.size(0)
    #free up memory
    del model, dataset, dataloader, classes, classes_dict, classes_all, classes_all_features, image, image_features, similarity, values, indices
    
    print(f"Model {model_name} - {pretrained} Accuracy: {correct/total}")

if __name__ == "__main__":
    for item in open_clip.list_pretrained():
        model_name = item[0]
        pretrained = item[1]

        accuracy(model_name, pretrained)