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
import sklearn.metrics

#set seed
torch.manual_seed(0)
np.random.seed(0)

#generate random vector
# random_vector = torch.randn(1, 10)
# print(random_vector)
# assert False

def compute_eer_auc(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    auc = sklearn.metrics.auc(fpr, tpr)
    return eer, auc

print("Starting Program")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "RN50"
pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption ClipLoss/checkpoints/epoch_50.pt"
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
    class_all.append(coco_test.coco.cats[i]['name'])
    class_all_index[coco_test.coco.cats[i]['name']] = i

#iterate through the test dataloader
total_acc = 0
count = 0
not_found_count = 0
model.eval()
score = []
ground_truth = []
with torch.no_grad():
    for i, (img, target) in enumerate(tqdm(coco_test)):
        img = img.to(device).unsqueeze(0)
        
        class_name = []
        
        #get class name
        for item in target:
            class_name.append(coco_test.coco.cats[item['category_id']]['name'])
        class_name = list(set(class_name))

        if(len(class_name) == 0):
            not_found_count += 1
            continue

        #create positive template from class name
        template_pos = ". ".join(class_name)
        text_features_pos = open_clip.tokenize(template_pos).to(device)
        text_features_pos = model.encode_text(text_features_pos)
        text_features_pos /= text_features_pos.norm(dim=-1, keepdim=True)
        text_features_pos = text_features_pos.to(device)
        
        #create negative template from random class name (not in the class name)
        class_name_neg = []
        while len(class_name_neg) < len(class_name):
            c = np.random.choice(class_all)
            if c not in class_name and c not in class_name_neg:
                class_name_neg.append(c)

        template_neg = ". ".join(class_name_neg)
        text_features_neg = open_clip.tokenize(template_neg).to(device)
        text_features_neg = model.encode_text(text_features_neg)
        text_features_neg /= text_features_neg.norm(dim=-1, keepdim=True)
        text_features_neg = text_features_neg.to(device)

        img_features = model.encode_image(img)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        
        pos_score = (img_features @ text_features_pos.T).squeeze(0)
        pos_score = pos_score.cpu().numpy()

        neg_score = (img_features @ text_features_neg.T).squeeze(0)
        neg_score = neg_score.cpu().numpy()

        
        score.append(pos_score)
        ground_truth.append(1)
        score.append(neg_score)
        ground_truth.append(0)
        
    score = np.array(score).reshape(-1)
    ground_truth = np.array(ground_truth).reshape(-1)
    eer, auc = compute_eer_auc(ground_truth, score)
    
    print("EER: ", eer)
    print("AUC: ", auc)
    print(score.shape)
    print(ground_truth.shape)
    print(not_found_count)
