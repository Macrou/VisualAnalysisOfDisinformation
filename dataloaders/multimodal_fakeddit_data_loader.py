
import pandas as pd
from dataloaders.fakeddit_data_loader import *

class Multimodal_Fakeddit(Fakeddit):
    def __init__(self,annotations_file,tokenizer ,img_dir="dataset/public_image_set",
                 transform=None, target_transform=None ):
        super().__init__(annotations_file,img_dir,transform,target_transform)
        self.tokenizer = tokenizer
        annotations = pd.read_csv(annotations_file,sep='\t')
        self.titles = annotations['clean_title']
        
        
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
        title = self.titles[idx]
        title_tokens = self.tokenizer(title)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return {
            'image':image, 
            'label':label,
            
        }