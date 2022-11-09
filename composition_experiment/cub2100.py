import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        image_attributes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
                                        sep=' ', names=['img_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.image_attributes = image_attributes

        #image attribute start from 1, so shift to 0
        self.image_attributes.attribute_id = self.image_attributes.attribute_id - 1
        print("Line 44")
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            print(e)
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_attributes = self.image_attributes[self.image_attributes.img_id == sample.img_id]
        img_attributes = img_attributes[img_attributes.is_present == 1].attribute_id.values.tolist()
        
        if self.transform is not None:
            img = self.transform(img)
        
        #padding
        for i in range(73-len(img_attributes)):
            img_attributes.append(-1)

        img_attributes = torch.tensor(img_attributes)

        return img, target, img_attributes

if __name__ == '__main__':
    root = "/scratch/mp5847/CUB_200_2011/"
    image_attributes = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'attributes', 'image_attribute_labels.txt'),
                                        sep=' ', names=['img_id', 'attribute_id', 'is_present', 'certainty_id', 'time'])

    #iterate through each row
    for index, row in tqdm(image_attributes.iterrows(), total=image_attributes.shape[0]):
        #randomly flip the is_present value
        image_attributes.loc[index, 'is_present'] = np.random.randint(0,2)

    #save the new file
    image_attributes.to_csv(os.path.join(root, 'CUB_200_2011', 'attributes', 'image_attribute_labels_random.txt'), sep=' ', index=False, header=False)