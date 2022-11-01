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
# device = torch.device("cpu")
model_name = "ViT-B-16"
pretrained = "laion400m_e32"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()

path2data_train = "/coco/train2017"
path2json_train = "/coco/annotations/instances_train2017.json"
path2data_test = "/coco/val2017"
path2json_test = "/coco/annotations/instances_val2017.json"

coco_train = dset.CocoDetection(root = path2data_train,
                                annFile = path2json_train, transform = preprocess)

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
            out = compose_net(target_emb).squeeze(1) 
            
            #calculate the loss
            loss = loss_fn(img_emb, out, torch.ones(img_emb.shape[0]).to(device)) 

            total_loss += loss.item()

    return total_loss / len(test_loader)

#define coco_custom dataset
class coco_custom(dset.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        super(coco_custom, self).__init__(root, annFile, transform, target_transform)
        self.coco = self.coco
        self.ids = self.ids

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index):
        img, target = super(coco_custom, self).__getitem__(index)
        
        while len(target) == 0:
            index = np.random.randint(0, len(self))
            img, target = super(coco_custom, self).__getitem__(index)

        target_all = []

        for item in target:
            target_all.append(torch.LongTensor([item["category_id"]])) #convert to torch tensor and move to device
        
        for i in range(95 - len(target_all)):
            target_all.append(torch.LongTensor([-1]))
           
        return img, torch.Tensor(target_all)



#create a custom dataset
class coco_precompute(dset.CocoDetection):
    def __init__(self, image_emb_path, target_emb_path, root, annFile, transform=None, target_transform=None):
        super(coco_precompute, self).__init__(root, annFile, transform, target_transform)
        self.image_emb = np.load(image_emb_path, allow_pickle=True)
        self.target_emb = np.load(target_emb_path,  allow_pickle=True)
        
    def __len__(self):
        return len(self.image_emb)

    def __getitem__(self, index):
        _, target = super(coco_precompute, self).__getitem__(index)
                   
        target_all = []

        for item in target:
            target_all.append(torch.LongTensor([item["category_id"]])) #convert to torch tensor and move to device
        
        while len(target_all) == 0:
            index = np.random.randint(0, len(self))
            _, target = super(coco_precompute, self).__getitem__(index)
            
            target_all = []

            for item in target:
                target_all.append(torch.LongTensor([item["category_id"]])) #convert to torch tensor and move to device

        #remove duplicates
        target_all = list(set(target_all)) #convert to torch tensor and move to device

        for i in range(95 - len(target_all)):
            target_all.append(torch.LongTensor([-1]))

        img_e = self.image_emb[index]

        #convert to torch tensor
        img_e = torch.from_numpy(img_e) 

        return img_e, torch.Tensor(target_all)

class CompositionNetwork(nn.Module):
    def __init__(self, n_emb, repr_size, zero_init=False):
        super().__init__()
        self.emb = nn.Embedding(n_emb, repr_size)
        if zero_init:
            self.emb.weight.data.zero_()

    def forward(self, target):
        result = []
        for item in target:
            result_item = [self.emb(torch.LongTensor([int(item[i].item())]).to(device)) for i in range(item.shape[0]) if item[i] != -1]
            try:
                result.append(torch.stack(result_item).sum(dim=0))
            except Exception as e:
                print(e)
                print(target)
                raise e
        result = torch.stack(result, dim=0) 

        return result

# coco_trainset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_img_emb.npy", 
#                             target_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_train_coco_target.npy", root=path2data_train, annFile=path2json_train, transform=preprocess)

coco_testset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_img_emb.npy", 
                            target_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_coco_target.npy", root=path2data_train, annFile=path2json_train, transform=preprocess)
                
# coco_trainloader = torch.utils.data.DataLoader(coco_trainset, batch_size=1024, shuffle=True, num_workers=5)

coco_testloader = torch.utils.data.DataLoader(coco_testset, batch_size=256, shuffle=False, num_workers=5)

rep_length = coco_testset[0][0].shape[0]
print("max class_all", max(list(coco_train.coco.cats.keys())))
compose_net = CompositionNetwork(max(list(coco_train.coco.cats.keys()))+1, rep_length) # 80
compose_net.to(device) # move the model parameters to CPU/GPU

print("Length class", len(class_all))
#consine similarity loss
loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean') #reduction='sum' by default

# optimizer Adam
optimizer = torch.optim.Adam(compose_net.parameters(), lr=0.01) #0.0001

wandb.init(project="composition_experiment", entity="mnphamx1", name=f"TRE {model_name} {pretrained} correct correct") #initialize wandb

print("Beginning Training")

num_epoch = 50
for epoch in range(num_epoch):
    total_loss = 0
    for i, (img_emb, target) in enumerate(coco_testloader):
        optimizer.zero_grad() #zero the gradient buffers
        img_emb = img_emb.to(device) #move to device

        out = compose_net(target).squeeze(1)

        loss = loss_fn(img_emb, out, torch.ones(1).to(device)) #b x 512
        loss.backward() #backpropagation
        optimizer.step() #does the update
        total_loss += loss.item() #add the loss to the total loss
        if i % 10 == 0:
            print(f"Epoch: {epoch} | Iteration: {i}/{len(coco_testloader)} | Loss: {total_loss / (i+1)}")
            wandb.log({"Loss": total_loss / (i+1)}) #log the loss to wandb  
