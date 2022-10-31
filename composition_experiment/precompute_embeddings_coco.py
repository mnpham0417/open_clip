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
from torch.utils.data import Dataset

print("Starting Program")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()

path2data_train = "/coco/train2017"
path2json_train = "/coco/annotations/instances_train2017.json"
path2data_test = "/coco/val2017"
path2json_test = "/coco/annotations/instances_val2017.json"

coco_train = dset.CocoDetection(root = path2data_train,
                                annFile = path2json_train)

print("Length of coco_train: ", len(coco_train))

class_all = []
for i in coco_train.coco.cats.keys():
    class_all.append(coco_train.coco.cats[i]['name'])

def test(compose_net, test_loader, device):
    total_loss = 0
    compose_net.eval()
    with torch.no_grad():
        for i, (img, class_all_features_cp) in enumerate(test_loader):
            img = img.to(device).squeeze(1) 
            class_all_features_cp = class_all_features_cp.to(device)

            with torch.no_grad():
                img_features = model.encode_image(img) 
                img_features /= img_features.norm(dim=-1, keepdim=True) 

            #get the output of the composition network
            out = compose_net(class_all_features_cp).squeeze(1) 

            #calculate the loss
            loss = loss_fn(img_features, out, torch.ones(img_features.shape[0]).to(device))

            total_loss += loss.item()

    return total_loss / (i + 1)

class_all_features = list()
class_all_features_index = {}
for i, c in enumerate(class_all):
    text = open_clip.tokenize(["a photo of " + c]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True) 
    text_features = text_features.cpu()
    #add 0 to the end of the vector
    class_all_features.append(text_features.detach().numpy()) 

    #convert class_all_features to torch tensor
    class_all_features_index[c] = i
class_all_features = torch.from_numpy(np.array(class_all_features)) 

#define coco_custom dataset
class coco_custom(dset.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(coco_custom, self).__init__(root, annFile, transform, target_transform, transforms)
        self.class_all_features = class_all_features
        self.coco = self.coco
        self.ids = self.ids

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index):
        img, target = super(coco_custom, self).__getitem__(index)
        img = preprocess(img).unsqueeze(0) 
        
        #copy class_all_features
        class_all_features_cp = self.class_all_features.clone()

        #set last element of the vector to 1 for the class corresponding to the category id
        for c in class_all:
            if(c not in target):
                class_all_features_cp[class_all_features_index[c]][:,:] = 0
        
        return img, class_all_features_cp.view(-1)

coco_trainset = coco_custom(root = path2data_train,
                                annFile = path2json_train)

coco_trainloader = torch.utils.data.DataLoader(coco_trainset, batch_size=256, shuffle=False, num_workers=5)

coco_testset = coco_custom(root = path2data_test,
                                annFile = path2json_test)
coco_testloader = torch.utils.data.DataLoader(coco_testset, batch_size=256, shuffle=False, num_workers=5)

print("Beginning Extracting Train Features")
train_img_emb_all = []
train_class_emb_all = []
with torch.no_grad():
    for i, (img_emb, target_emb) in enumerate(tqdm(coco_trainloader)):
            
        img_emb = img_emb.to(device).squeeze(1)
        target_emb = target_emb.to(device) 

        img_emb = model.encode_image(img_emb) 
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        train_img_emb_all.extend(img_emb.cpu().numpy()) 
        train_class_emb_all.extend(target_emb.cpu().numpy()) 
        
train_img_emb_all = np.array(train_img_emb_all) 
train_class_emb_all = np.array(train_class_emb_all) 

#save the embeddings
np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_img_emb.npy", train_img_emb_all) 
np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_target.npy", train_class_emb_all) 

print("Beginning Extracting Test Features")

test_img_emb_all = []
test_class_emb_all = []
with torch.no_grad():
    for i, (img_emb, target_emb) in enumerate(tqdm(coco_testloader)):
            
        img_emb = img_emb.to(device).squeeze(1)
        target_emb = target_emb.to(device) 

        img_emb = model.encode_image(img_emb) 
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        test_img_emb_all.extend(img_emb.cpu().numpy()) 
        test_class_emb_all.extend(target_emb.cpu().numpy()) 

test_img_emb_all = np.array(test_img_emb_all)
test_class_emb_all = np.array(test_class_emb_all) 

#save the embeddings
np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_img_emb.npy", test_img_emb_all)
np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_target.npy", test_class_emb_all) # 10x513