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
from sklearn.metrics import classification_report

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
pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption TripletLoss(1 + 2 + 3 + 4) + ClipLoss - lambda=0/checkpoints/epoch_50.pt"
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
    class_all_index["a photo of " + coco_test.coco.cats[i]['name']] = i

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