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
model_name = "ViT-L-14"
pretrained = "laion2b_s32b_b82k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
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
text_features /= text_features.norm(dim=-1, keepdim=True) # 1x512
text_features = text_features.to(device) # 1x512

print("text_features.shape: ", text_features.shape) # 1x512

#iterate through the test dataloader
# total_acc = 0
# count = 0
# not_found_count = 0
# model.eval()
# with torch.no_grad():
#     for i, (img, target) in enumerate(tqdm(coco_test)):
#         img = img.to(device).unsqueeze(0)
        
#         class_name = []
    
#         #get class name
#         for item in target:
#             class_name.append("a photo of " + coco_test.coco.cats[item['category_id']]['name']) #get the class name
#         class_name = list(set(class_name))

#         img_features = model.encode_image(img) # 1x512
#         img_features /= img_features.norm(dim=-1, keepdim=True) # 1x512
        
#         pred = (img_features @ text_features.T).squeeze(0) # 1x80
        
#         #get top len(class_name) predictions
#         top_pred = torch.topk(pred, len(class_name))[1] # 1xlen(class_name)
#         top_pred_class = [class_all[i] for i in top_pred] # 1xlen(class_name)

#         print("top_pred_class: ", top_pred_class) # 1xlen(class_name)

#         try:
#             accuracy_score = accuracy(top_pred_class, class_name) # 1x1
#             count += 1
#         except:
#             not_found_count += 1
#         total_acc += accuracy_score
    
#     print("Average accuracy: ", total_acc / count) # 1x1
#     print("Not found count: ", not_found_count) # 1x1

total_acc = 0
count = 0
not_found_count = 0
model.eval()
with torch.no_grad():
    for i, (img, target) in enumerate(tqdm(coco_test)):
        img = img.to(device).unsqueeze(0)
        
        class_name = []
    
        #get class name
        for item in target:
            class_name.append("a photo of " + coco_test.coco.cats[item['category_id']]['name']) #get the class name
        class_name = list(set(class_name))

        img_features = model.encode_image(img) # 1x512
        img_features /= img_features.norm(dim=-1, keepdim=True) # 1x512
        
        pred = (img_features @ text_features.T).squeeze(0) # 1x80
        
        #get top len(class_name) predictions
        top_pred = torch.topk(pred, len(class_name))[1] # 1xlen(class_name)
        top_pred_class = [class_all[i] for i in top_pred] # 1xlen(class_name)

        try:
            accuracy_score = accuracy(top_pred_class, class_name) # 1x1
            count += 1
        except:
            not_found_count += 1
        if(accuracy_score == 1):
            total_acc += 1
        # total_acc += accuracy_score
    
    print("Average accuracy: ", total_acc / count) # 1x1
    print("Not found count: ", not_found_count) # 1x1