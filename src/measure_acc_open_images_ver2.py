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
import sklearn.metrics

#set seed
torch.manual_seed(0)
np.random.seed(0)

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
model_name = "ViT-L-14"
# pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption TripletLoss(2 + 3 + 4) + ClipLoss - lambda=10/checkpoints/epoch_50.pt"
pretrained = "laion2b_s32b_b82k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
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
    class_all.append(class_name.lower())
    class_all_index[class_name.lower()] = i
    
text = open_clip.tokenize(class_all).to(device)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True) # 1x512
text_features = text_features.to(device) # 1x512

#iterate through the test dataloader
total_acc = 0
count = 0
not_found_count = 0
model.eval()
score = []
ground_truth = []
with torch.no_grad():
    for image_path in tqdm(glob.glob("/open-images-dataset/test/*.jpg")):
        if(count == 200):
            break
        
        img_id = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        img_classes = open_images_test_labels[open_images_test_labels['ImageID'] == img_id]['LabelName'].values
        img_classes = [open_images_class_descriptions_boxable[open_images_class_descriptions_boxable["LabelName"] == c].iloc[0]['DisplayName'] for c in img_classes]

        class_name = []
        #get class name
        for c in img_classes:
            class_name.append(c.lower()) #get the class name
        class_name = list(set(class_name))

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

        # print("Positive Template: ", template_pos)
        # print("Negative Template: ", template_neg)
        # assert False

        img_features = model.encode_image(image)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        
        pos_score = (img_features @ text_features_pos.T).squeeze(0)
        pos_score = pos_score.cpu().numpy()

        neg_score = (img_features @ text_features_neg.T).squeeze(0)
        neg_score = neg_score.cpu().numpy()

        
        score.append(pos_score)
        ground_truth.append(1)
        score.append(neg_score)
        ground_truth.append(0)
        count += 1

    score = np.array(score).reshape(-1)
    ground_truth = np.array(ground_truth).reshape(-1)
    eer, auc = compute_eer_auc(ground_truth, score)
    
    print("EER: ", eer)
    print("AUC: ", auc)
    print(score.shape)
    print(ground_truth.shape)
    print(not_found_count)