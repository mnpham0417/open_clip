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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model_name = "RN50"
pretrained = "/scratch/mp5847/open-clip/logs/CLIP RN50 COCO Caption TripletLoss(1 + 2 + 3 + 4) + ClipLoss - lambda=10/checkpoints/epoch_50.pt"
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

#create a custom dataset
class coco_precompute(dset.CocoDetection):
    def __init__(self, root, annFile, image_emb_path):
        super(coco_precompute, self).__init__(root, annFile)
        self.image_emb = np.load(image_emb_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.image_emb)

    def __getitem__(self, index):
        _, target = super().__getitem__(index)
        img_e = self.image_emb[index] # 1x512
        if(len(target) == 0):
            img_e, img_classes = self.__getitem__(np.random.randint(0, len(self.image_emb)))
        else:
            img_classes = []
            for i in target:
                img_classes.append(i['category_id'])

            img_classes = list(set(img_classes))

            #pad img_classes with -1 to make it 30
            if len(img_classes) < 30:
                img_classes = img_classes + [-1]*(30-len(img_classes))

            #convert to torch tensor
            img_e = torch.from_numpy(img_e) # 1x512
            img_classes = torch.LongTensor(img_classes) # 30

        return img_e, img_classes

class CompositionNetwork(nn.Module):
    def __init__(self, n_emb, repr_size, zero_init=False):
        super().__init__()
        self.emb = nn.Embedding(n_emb, repr_size)
        self.linear1 = nn.Linear(repr_size, repr_size)
        self.linear2 = nn.Linear(repr_size, repr_size)
        if zero_init:
            self.emb.weight.data.zero_()

    def forward(self, target):
        result = []
        for item in target:
            result_item = [self.emb(torch.LongTensor([int(item[i].item())]).to(device)) for i in range(item.shape[0]) if item[i] != -1]
            try:
                input_ = torch.stack(result_item).mean(dim=0)
                input_ = nn.functional.relu(self.linear1(input_))
                input_ = nn.functional.relu(self.linear2(input_))
                result.append(input_)
                # result.append(torch.stack(result_item).sum(dim=0))
            except Exception as e:
                print(e)
                print(target)
                raise e
        result = torch.stack(result, dim=0) 

        return result

coco_trainset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/RN50_COCO Caption TripletLoss(1 + 2 + 3 + 4) + ClipLoss - lambda=10_train_coco_img_emb.npy", 
                            root = path2data_train,
                            annFile = path2json_train)

coco_trainloader = torch.utils.data.DataLoader(coco_trainset, batch_size=1024, shuffle=True, num_workers=5)

coco_testset = coco_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/RN50_COCO Caption TripletLoss(1 + 2 + 3 + 4) + ClipLoss - lambda=10_test_coco_img_emb.npy",
                            root = path2data_test,
                            annFile = path2json_test)
                            
coco_testloader = torch.utils.data.DataLoader(coco_testset, batch_size=1024, shuffle=True, num_workers=5)


compose_net = CompositionNetwork(91, 1024) #there are 80 classes in coco dataset
compose_net.to(device) # move the model parameters to CPU/GPU

#consine similarity loss
loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean') #reduction='sum' by default

#optimizer Adam
optimizer = torch.optim.Adam(compose_net.parameters(), lr=0.01) #0.0001

wandb.init(project="composition_experiment", entity="mnphamx1", name=f"Non-linear TRE RN50 COCO Caption TripletLoss(1 + 2 + 3 + 4) + ClipLoss") #initialize wandb

print("Beginning Training")
#iterate through all images in the dataset
num_epoch = 50
for epoch in range(num_epoch):

    total_loss = 0
    for i, (img_emb, target) in enumerate(coco_trainloader):

        optimizer.zero_grad() #zero the gradient buffers
        img_emb = img_emb.to(device) #move to device

        out = compose_net(target).squeeze(1)

        loss = loss_fn(img_emb, out, torch.ones(1).to(device)) 
        loss.backward() #backpropagation
        optimizer.step() #does the update
        total_loss += loss.item() #add the loss to the total loss
        if i % 5 == 0:
            print(f"Epoch: {epoch} | Iteration: {i}/{len(coco_trainloader)} | Loss: {total_loss / (i+1)}")
            wandb.log({"Loss": total_loss / (i+1)}) #log the loss to wandb  

test_loss = test(compose_net, coco_testloader, device)
print(f"Test Loss: {test_loss}") # 1x1
wandb.log({"Test Loss": test_loss}) # 1x1