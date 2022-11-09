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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model_name = "RN50-quickgelu"
pretrained = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
model.eval()

#create a custom dataset
class cub_precompute(Cub2011):
    def __init__(self,  image_emb_path, root = "/scratch/mp5847/CUB_200_2011/", train = True, transform = None, download=False):
        super(cub_precompute, self).__init__(root, train, transform, download)
        self.image_emb = np.load(image_emb_path, allow_pickle=True)
        
    def __len__(self):
        return len(self.image_emb)

    def __getitem__(self, index):
        _, _, img_attributes = super(cub_precompute, self).__getitem__(index)

        img_e = self.image_emb[index]

        #convert to torch tensor
        img_e = torch.from_numpy(img_e) 

        return img_e, img_attributes

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

cub_test = cub_precompute(image_emb_path = f"/scratch/mp5847/precomputed_embeddings_comp_exp/{model_name}_{pretrained}_test_cub_img_emb.npy",
                        root = "/scratch/mp5847/CUB_200_2011/", train = False, transform = None, download=False)

cub_testloader = torch.utils.data.DataLoader(cub_test, batch_size=256, shuffle=True, num_workers=0)

rep_length = cub_test[0][0].shape[0]

compose_net = CompositionNetwork(322, rep_length) #there are 322 binary attributes
compose_net.to(device) # move the model parameters to CPU/GPU

#consine similarity loss
loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean') #reduction='sum' by default

# optimizer Adam
optimizer = torch.optim.Adam(compose_net.parameters(), lr=0.01) #0.0001

wandb.init(project="composition_experiment", entity="mnphamx1", name=f"TRE CUB {model_name} {pretrained}") #initialize wandb

print("Beginning Training")

num_epoch = 50
for epoch in range(num_epoch):
    total_loss = 0
    for i, (img_emb, target) in enumerate(cub_testloader):
        optimizer.zero_grad() #zero the gradient buffers
        img_emb = img_emb.to(device) #move to device

        out = compose_net(target).squeeze(1)

        loss = loss_fn(img_emb, out, torch.ones(1).to(device)) #b x 512
        loss.backward() #backpropagation
        optimizer.step() #does the update
        total_loss += loss.item() #add the loss to the total loss
        if i % 5 == 0:
            print(f"Epoch: {epoch} | Iteration: {i}/{len(cub_testloader)} | Loss: {total_loss / (i+1)}")
            wandb.log({"Loss": total_loss / (i+1)}) #log the loss to wandb  