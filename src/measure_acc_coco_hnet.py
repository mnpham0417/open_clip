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
pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption HNET x 2/checkpoints/epoch_117.pt"
pretrained_hnet = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption HNET x 2/checkpoints/hnet_epoch_117.pt"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
hnet = Mike_net_linear(1024)
hnet.load_state_dict(torch.load(pretrained_hnet))
hnet.to(device)
hnet.eval()
model.eval()

path2data_train = "/coco/train2017"
path2json_train = "/coco/annotations/instances_train2017.json"
path2data_test = "/coco/val2017"
path2json_test = "/coco/annotations/instances_val2017.json"

coco_test = dset.CocoDetection(root = path2data_test,
                                annFile = path2json_test, transform=preprocess)

class_all = []
class_all_index = {}
for i in coco_test.coco.cats.keys():
    class_all.append("a photo of " + coco_test.coco.cats[i]['name'])
    class_all_index[coco_test.coco.cats[i]['name']] = i

text = open_clip.tokenize(class_all).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True) 
text_features = text_features.to(device) 

#iterate through the test dataloader
total_acc = 0
count = 0
not_found_count = 0

with torch.no_grad():
    for i, (img, target) in enumerate(tqdm(coco_test)):
        img = img.to(device).unsqueeze(0)
        
        class_name = []
        #get class name
        for item in target:
            class_name.append("a photo of " + coco_test.coco.cats[item['category_id']]['name']) #get the class name
        class_name = list(set(class_name))

        img_features = model.encode_image(img) # 
        img_features /= img_features.norm(dim=-1, keepdim=True) 
        
        #repeat img_features
        img_features = img_features.repeat(text_features.shape[0], 1)

        score_hnet = hnet(img_features, text_features).squeeze()

        #get top len(class_name) predictions
        top_pred = torch.topk(score_hnet, len(class_name))[1] # 1xlen(class_name)
        top_pred_class = [class_all[i] for i in top_pred] # 1xlen(class_name)
        try:
            accuracy_score = accuracy(top_pred_class, class_name) 
            count += 1
        except:
            not_found_count += 1
        total_acc += accuracy_score
    
    print("Average accuracy: ", total_acc / count) # 1x1
    print("Not found count: ", not_found_count) # 1x1