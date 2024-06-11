import pandas as pd
from PIL import Image
import os
import logging as log
import matplotlib.pyplot as plt

def cleanData(annotations_file,new_annotations_file ,img_dir="dataset/public_image_set"):
    error_count = 0
    annotations = pd.read_csv(annotations_file,sep='\t',index_col='id')
    ids = annotations['id'].values
    for id in ids:
        path = os.path.join(img_dir,id)
        try:
           img = Image.open(path)
           img.verify()
        except (IOError, SyntaxError) as e:
            log.error(f"Error loading image {path}: {e}")
            annotations.drop(index=id)
            error_count += 1
    log.info("Saving data into new CSV")
    log.info(f"Removed {error_count} images")
    annotations.to_csv(new_annotations_file,sep='\t')

def EDA(annotations_file,img_dir="dataset/public_image_set"):
    annotations = pd.read_csv(annotations_file,sep='\t',index_col='id')
    label_count = annotations['2_way_label'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(label_count.index,quality_counts)
    plt.title('Count of 2 way label')
    plt.xlabel('2 ') 
    plt.ylabel('Count')


if __name__ == "__main__":
    cleanData()