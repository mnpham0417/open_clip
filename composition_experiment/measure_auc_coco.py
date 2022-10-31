import torchvision.datasets as dset
import open_clip
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def get_scores(model, preprocess, img, class_list):
    with torch.no_grad():
        #get all classes other than the ones in the image
        class_negative = []
        for i in class_all:
            if i not in class_list:
                class_negative.append(i)

        #sample len(class_list) classes from class_all not in class_list
        class_negative = np.random.choice(class_all, len(class_list), replace=False)
        class_negative = [i for i in class_negative if i not in class_list]

        image = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        ground_truth = []

        scores = []

        for c in class_list:
            text_features = class_all_features["a photo of " + c]
            scores.append((image_features @ text_features.T).item())
            ground_truth.append(1)

        for c in class_negative:
            text_features = class_all_features["a photo of " + c]
            scores.append((image_features @ text_features.T).item())
            ground_truth.append(0)

    return scores, ground_truth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model_name = "RN50x4"
pretrained = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)

path2data="/data/mp5847_dataset/coco/train2017"
path2json="/data/mp5847_dataset/coco/annotations/instances_train2017.json"
coco_train = dset.CocoDetection(root = path2data,
                                annFile = path2json)

#get all classes in the dataset
class_all = []
for i in coco_train.coco.cats.keys():
    class_all.append(coco_train.coco.cats[i]['name'])

class_all_features = {}
for c in class_all:
    text = open_clip.tokenize(["a photo of " + c]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    class_all_features["a photo of " + c] = text_features

count = 5000
final_scores = []
final_ground_truth = []
for i, (img, target) in enumerate(tqdm(coco_train)):
    if i == count:
        break
    #get the class corresponding to the category id
    class_list = []
    for i in range(len(target)):
        class_list.append(coco_train.coco.cats[target[i]['category_id']]['name'])
    class_list = list(set(class_list))

    scores, ground_truth = get_scores(model, preprocess, img, class_list)
    final_scores.extend(scores)
    final_ground_truth.extend(ground_truth)

#calculate AUC

#flatten final_scores
print(np.array(final_scores).shape, len(final_ground_truth))
auc = roc_auc_score(final_ground_truth, final_scores)
print(model_name, pretrained ,auc)














