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
from cub2100 import Cub2011

print("Starting Program")

with torch.no_grad():
    for item in open_clip.list_pretrained():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = item[0]
        pretrained = item[1]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()

        dataset_path = '/scratch/mp5847/CUB_200_2011/'
        cub_train = Cub2011(root=dataset_path, train=True, download=False, transform=preprocess)
        cub_trainloader = torch.utils.data.DataLoader(cub_train, batch_size=256, shuffle=False, num_workers=5)

        cub_test = Cub2011(root=dataset_path, train=False, download=False, transform=preprocess)
        cub_testloader = torch.utils.data.DataLoader(cub_test, batch_size=256, shuffle=False, num_workers=5)

        print("Beginning Extracting Train Features")
        train_img_emb_all = []
        train_class_emb_all = []
        with torch.no_grad():
            for i, (img_emb, target_emb, _) in enumerate(tqdm(cub_trainloader)):
                    
                img_emb = img_emb.to(device).squeeze(1)
                target_emb = target_emb.to(device) 

                img_emb = model.encode_image(img_emb) 
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                train_img_emb_all.extend(img_emb.cpu().numpy()) 
                train_class_emb_all.extend(target_emb.cpu().numpy()) 
                
        train_img_emb_all = np.array(train_img_emb_all) 
        train_class_emb_all = np.array(train_class_emb_all) 

        #save the embeddings
        np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_cub_img_emb.npy", train_img_emb_all) 
        # np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_cub_target.npy", train_class_emb_all) 

        print("Beginning Extracting Test Features")

        test_img_emb_all = []
        test_class_emb_all = []
        with torch.no_grad():
            for i, (img_emb, target_emb, _) in enumerate(tqdm(cub_testloader)):
                    
                img_emb = img_emb.to(device).squeeze(1)
                target_emb = target_emb.to(device) 

                img_emb = model.encode_image(img_emb) 
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                test_img_emb_all.extend(img_emb.cpu().numpy()) 
                test_class_emb_all.extend(target_emb.cpu().numpy()) 

        test_img_emb_all = np.array(test_img_emb_all)
        test_class_emb_all = np.array(test_class_emb_all) 

        #save the embeddings
        np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_cub_img_emb.npy", test_img_emb_all)
        # np.save(f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_cub_target.npy", test_class_emb_all) # 10x513