import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image,UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None


class Fakeddit(Dataset):
    '''
    Dataset implementation in Pytorch of the r/Fakeddit Dataset by Nakamura, Kai and Levy, 
    Sharon and Wang, William Yang.
    '''
    def __init__(self,annotations_file, img_dir="dataset/public_image_set",
                 transform=None, target_transform=None):
        anotations = pd.read_csv(annotations_file,sep='\t')
        self.id = anotations['id'].values
        self.img_dir = img_dir
        self.img_labels = anotations['2_way_label'].values
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        '''
        The length of the dataset. 
        Returns
        -------
        The length of the dataset
        '''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''
        Gets one sample form the dataset

        Parameters
        ----------
        idx : int
            the index of the data in the dataset
        Returns
        -------
        the image and the label after transformation 
        '''
        name = self.id[idx] + '.jpg'
        path = os.path.join(self.img_dir,name)
        try:
            image  = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            image = Image.fromarray(np.zeros((224, 224, 3),
                                             dtype=np.uint8))  # Adjust dimensions as needed
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
