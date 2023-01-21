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
import glob
import pandas as pd
import os
from sklearn.metrics import classification_report
from training.h_net import *


def accuracy(pred, target):
    '''
    function that calculates intersection over union of pred and target
    @param pred: list of predicted classes
    @param target: list of target classes
    '''

    #intersection of pred and target
    intersection = set(pred).intersection(set(target))
    #union of pred and target
    union = set(pred).union(set(target))

    return len(intersection) / len(union)


print("Starting Program")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "RN50"
pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption HNET 200 epoch/checkpoints/epoch_200.pt"
pretrained_hnet = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption HNET 200 epoch/checkpoints/hnet_epoch_200.pt"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
hnet = Mike_net_linear(1024)
hnet.load_state_dict(torch.load(pretrained_hnet))
hnet.to(device)
hnet.eval()
model.eval()

open_images_class_descriptions_boxable = pd.read_csv('/scratch/mp5847/open_images_annotations/oidv7-class-descriptions-boxable.csv')
open_images_test_labels = pd.read_csv('/scratch/mp5847/open_images_annotations/Image labels/test-annotations-human-imagelabels-boxable.csv')

classes = open_images_class_descriptions_boxable['LabelName'].values

print("Number of classes: ", len(classes))

#to lower case
classes = [x.lower() for x in classes]

class_all = []
class_all_index = {}
for i, c in enumerate(classes):
    class_name = open_images_class_descriptions_boxable[open_images_class_descriptions_boxable["LabelName"] == c].iloc[0]['DisplayName']
    class_all.append("a photo of " + class_name.lower())
    class_all_index["a photo of " + class_name.lower()] = i
    
text = open_clip.tokenize(class_all).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True) # 1x512
text_features = text_features.to(device) # 1x512

#iterate through the test dataloader
total_acc = 0
count = 0
not_found_count = 0
model.eval()
pred_all = []
target_all = []
with torch.no_grad():
    for image_path in tqdm(glob.glob("/open-images-dataset/test/*.jpg")):
        if(count == 200):
            break
        
        img_id = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)

        img_features = model.encode_image(image)
        img_features /= img_features.norm(dim=-1, keepdim=True)

        #repeat img_features
        img_features = img_features.repeat(text_features.shape[0], 1)
        
        img_classes = open_images_test_labels[open_images_test_labels['ImageID'] == img_id]['LabelName'].values
        img_classes = [open_images_class_descriptions_boxable[open_images_class_descriptions_boxable["LabelName"] == c].iloc[0]['DisplayName'] for c in img_classes]

        class_name = []
        #get class name
        for c in img_classes:
            class_name.append("a photo of " + c.lower()) #get the class name
        class_name = list(set(class_name))

        score_hnet = hnet(img_features, text_features).squeeze()
        
        top_pred = torch.topk(score_hnet, len(class_name))[1] # 1xlen(class_name)
        top_pred_class = [class_all[i] for i in top_pred] # 1xlen(class_name)

        # top_pred_class = []

        # #randomly select len(class_name) number of classes
        # for i in range(len(class_name)):
        #     top_pred_class.append(class_all[np.random.randint(0, len(class_all))])

        try:
            accuracy_score = accuracy(top_pred_class, class_name) # 1x1
            pred_all.extend([class_all_index[class_name] for class_name in top_pred_class])
            target_all.extend([class_all_index[class_name] for class_name in class_name])
            total_acc += accuracy_score
            count += 1
        except Exception as e:
            # print(e)
            not_found_count += 1
        
    print(model_name, pretrained)
    print(count, not_found_count)
    print("Average accuracy: ", total_acc / count) # 1x1
    print("Not found count: ", not_found_count) # 1x1
    print(classification_report(target_all, pred_all))