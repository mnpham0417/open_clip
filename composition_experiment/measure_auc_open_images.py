import torchvision.datasets as dset
import open_clip
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import glob
from PIL import Image

def get_scores(model, preprocess, img, class_list):
    with torch.no_grad():
        #get all classes other than the ones in the image
        class_negative = []
        for i in class_all:
            if i not in class_list:
                class_negative.append(i)

        image = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        ground_truth = []

        scores = []
        for c in class_list:
            text_features = class_all_features[c].to(device)
            scores.append((image_features @ text_features.T).item())
            ground_truth.append(1)

        for c in class_negative:
            text_features = class_all_features[c].to(device)
            scores.append((image_features @ text_features.T).item())
            ground_truth.append(0)

    return scores, ground_truth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ViT-B-16"
pretrained = "laion400m_e32"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)

test_labels_path = "/media/mnpham/HARD_DISK_3/open_images/test-annotations-human-imagelabels-boxable.csv"
test_labels = np.loadtxt(test_labels_path, dtype=str, delimiter=",", skiprows=1)
test_labels_dict = defaultdict(list)

class_description_path = "/media/mnpham/HARD_DISK_3/open_images/class-descriptions-boxable.csv"
class_description = np.loadtxt(class_description_path, dtype=str, delimiter=",", skiprows=0)
class_description_dict = dict(class_description)

print("class_description_dict", class_description_dict)

for item in test_labels:
    test_labels_dict[item[0]].append(class_description_dict[item[2]])

#convert to set
for key in test_labels_dict:
    test_labels_dict[key] = list(set(test_labels_dict[key]))

test_images_path = "/media/mnpham/HARD_DISK_3/open_images/test/"

test_images = glob.glob(test_images_path + "*.jpg")

#shuffle the images
np.random.shuffle(test_images)

count = 5000
test_images = test_images[:count]

#get all class
class_all = []
for key in test_labels_dict:
    class_all.extend(test_labels_dict[key])
class_all = list(set(class_all))

class_all_features = {}
for c in class_all:
    text = open_clip.tokenize([c]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    class_all_features[c] = text_features.detach().cpu().numpy()

final_scores = []
final_ground_truth = []
for i, img_path in enumerate(tqdm(test_images)):
    class_list = test_labels_dict[img_path.split("/")[-1].split(".")[0]]
    img = Image.open(img_path)
    scores, ground_truth = get_scores(model, preprocess, img, class_list)
    final_scores.extend(scores)
    final_ground_truth.extend(ground_truth)

#calculate AUC

auc = roc_auc_score(final_ground_truth, final_scores)
print(model_name, pretrained ,auc)

    


