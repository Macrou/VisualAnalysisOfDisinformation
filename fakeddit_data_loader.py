import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image,UnidentifiedImageError


class Fakeddit(Dataset):
    '''
    Dataset implementation of the Fakeddit by Nakamura, Kai and Levy, Sharon and Wang, William Yang.
    '''

    def __init__(self,annotations_file, img_dir="dataset/public_image_set", transform=None,
                 target_transform=None):
        anotations = pd.read_csv(annotations_file,sep='\t')
        self.id = anotations['id'].values
        self.img_dir = img_dir
        self.img_labels = anotations['2_way_label'].values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        Gets the data from the.
        '''
        name = self.id[idx] + '.jpg'
        path = os.path.join(self.img_dir,name)
        try:
            image  = Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError: Cannot identify image file {path}")
            image = Image.fromarray(np.zeros((224, 224, 3),
                                             dtype=np.uint8))  # Adjust dimensions as needed
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
