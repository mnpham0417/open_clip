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
model_name = "ViT-g-14"
pretrained = "laion2b_s12b_b42k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()

path2data_train = "/coco/train2017"
path2json_train = "/coco/annotations/instances_train2017.json"
path2data_test = "/coco/val2017"
path2json_test = "/coco/annotations/instances_val2017.json"

coco_train = dset.CocoDetection(root = path2data_train,
                                annFile = path2json_train)

coco_test = dset.CocoDetection(root = path2data_test,
                                annFile = path2json_test)
print("Length of coco_train: ", len(coco_train))
print("Length of coco_test: ", len(coco_test))


class_all = []
for i in coco_train.coco.cats.keys():
    class_all.append(coco_train.coco.cats[i]['name'])

class_all_test = []
for i in coco_test.coco.cats.keys():
    class_all_test.append(coco_test.coco.cats[i]['name'])

assert class_all == class_all_test

def test(compose_net, test_loader, device):
    total_loss = 0
    compose_net.eval()
    with torch.no_grad():
        for i, (img_emb, target_emb) in enumerate(test_loader):
            
            img_emb = img_emb.to(device)
            target_emb = target_emb.to(device) 

            #get the output of the composition network
            out = compose_net(target_emb).squeeze(1) # 10x512
            
            #calculate the loss
            loss = loss_fn(img_emb, out, torch.ones(img_emb.shape[0]).to(device)) # 10x1

            total_loss += loss.item()

    return total_loss / len(test_loader)

# class_all_features = list()
# class_all_features_index = {}
# for i, c in enumerate(class_all):
#     text = open_clip.tokenize(["a photo of " + c]).to(device)
#     text_features = model.encode_text(text)
#     text_features /= text_features.norm(dim=-1, keepdim=True) # 1x512
#     text_features = text_features.cpu()
#     #add 0 to the end of the vector
#     text_features_concat = torch.cat((text_features, torch.zeros(1, 1)), dim=1) # 1x513
#     class_all_features.append(text_features_concat.detach().numpy()) # 1x513

#     #convert class_all_features to torch tensor
#     class_all_features_index[c] = i
# class_all_features = torch.from_numpy(np.array(class_all_features)) # 80x513

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
        img = preprocess(img).unsqueeze(0) # 1x3x224x224
        
        #copy class_all_features
        class_all_features_cp = self.class_all_features.clone()
        # class_all_features_out = []

        #set last element of the vector to 1 for the class corresponding to the category id
        for i in range(len(target)):
            # print(class_all_features_cp[class_all_features_index[self.coco.cats[target[i]['category_id']]['name']]].shape)
            class_all_features_cp[class_all_features_index[self.coco.cats[target[i]['category_id']]['name']]][:,:] = 0
        
        return img, class_all_features_cp.view(-1)

#create a custom dataset
class coco_precompute(Dataset):
    def __init__(self, image_emb_path, target_emb_path):
        
        self.image_emb = np.load(image_emb_path, allow_pickle=True)
        self.target_emb = np.load(target_emb_path,  allow_pickle=True)
        
    # (5000, 513)
    def __len__(self):
        return len(self.image_emb)

    def __getitem__(self, index):
        img_e = self.image_emb[index] # 1x512
        target_e = self.target_emb[index] # 1x513

        #convert to torch tensor
        img_e = torch.from_numpy(img_e) # 1x512
        target_e = torch.from_numpy(target_e) # 1x513

        return img_e, target_e


#custom Network
class CompositionNetwork(torch.nn.Module):
    def __init__(self, in_len, out_len, device):
        super(CompositionNetwork, self).__init__()
        #80 linear layers, one for each class
        self.model = nn.Linear(in_len, out_len) #hard code for now

    def forward(self, class_features):

        out = self.model(class_features) #b x 512

        return out

coco_trainset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_img_emb.npy", 
                            target_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_target.npy")

coco_trainloader = torch.utils.data.DataLoader(coco_trainset, batch_size=1024, shuffle=True, num_workers=5)

coco_testset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_img_emb.npy", 
                            target_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_target.npy")

coco_testloader = torch.utils.data.DataLoader(coco_testset, batch_size=1024, shuffle=True, num_workers=5)

in_len = coco_trainset[0][1].shape[0]
out_len = coco_trainset[0][0].shape[0]

compose_net = CompositionNetwork(in_len, out_len, device) # 80
compose_net.to(device) # move the model parameters to CPU/GPU

#consine similarity loss
loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean') #reduction='sum' by default

#optimizer Adam
optimizer = torch.optim.Adam(compose_net.parameters(), lr=0.0001) #0.0001

wandb.init(project="composition_experiment", entity="mnphamx1", name=f"TRE {model_name} {pretrained} correct") #initialize wandb

print("Beginning Training")
#iterate through all images in the dataset
num_epoch = 20
for epoch in range(num_epoch):
    total_loss = 0
    for i, (img_emb, target_emb) in enumerate(coco_trainloader):
        
        compose_net.zero_grad() # reset the gradients
        img_emb = img_emb.to(device)
        target_emb = target_emb.to(device) 

        #get the output of the composition network
        out = compose_net(target_emb).squeeze(1) # 10x512
        
        #calculate the loss
        loss = loss_fn(img_emb, out, torch.ones(img_emb.shape[0]).to(device)) # 10x1
        loss.backward() # calculate the gradients
        optimizer.step() # update the weights

        total_loss += loss.item()

        #print statistics 
        if(i % 10 == 0):
            
            print(f"Epoch: {epoch} | Iteration: {i} / {len(coco_trainloader)} | Train Loss: {total_loss / (i + 1)}") # 1x1
            wandb.log({"Train Loss": total_loss / (i + 1)}) # 1x1

test_loss = test(compose_net, coco_testloader, device)
print(f"Test Loss: {test_loss}") # 1x1
wandb.log({"Test Loss": test_loss}) # 1x1