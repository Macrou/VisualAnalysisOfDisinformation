"""
This file cleans the data sets of faulty images,
and analyses the training set for corelation between title and label.
"""

import os
from collections import Counter

import pandas as pd
from PIL import Image,UnidentifiedImageError
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

Image.MAX_IMAGE_PIXELS = None

def clean_data(annotations_file,new_annotations_file ,img_dir="dataset/public_image_set"):
    """Cleans the data set of faulty images.Checks if there is an exception and
    if it exists it removes the image from the dataset.
    At the end it creates a new annotation file containing only valid images.

    Args:
        annotations_file (string): the annotation file of the data set
        new_annotations_file (string): the path for the new annotation file when created.
        img_dir (str, optional): The path to the image directory of the dataset.
        Defaults to "dataset/public_image_set".
    """
    error_count = 0
    annotations = pd.read_csv(annotations_file,sep='\t',index_col='id')
    indices = annotations.index.values
    for index in indices:
        name = index + '.jpg'
        path = os.path.join(img_dir,name)
        try:
            img = Image.open(path)
            img.verify()
        except (UnidentifiedImageError, OSError, IOError) as e:
            print(f"Error loading image {path}: {e}")
            annotations.drop(index=index,inplace=True)
            error_count += 1
    print("Saving data into new CSV")
    print(f"Removed {error_count} images")
    annotations.to_csv(new_annotations_file,sep='\t')

def label_distribution(df):
    """Plots the distribution of the 2_way_label

    Args:
        df (pd.DataFrame): the Data Frame used for this plot
    """
    df['2_way_label'].value_counts().plot(kind='bar')
    plt.title('Distribution of 2_way_label')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.savefig(fname='results/plots/labelFrequency.png',format='png')
    plt.clf()
def title_analyzer(df):
    """Plots the word cloud with the most common words in the clean titles of the dataset,
    and prints 10 of the most common words in the title

    Args:
        df (_type_): _description_
    """
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(' '.join(df['clean_title']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(fname='results/plots/wordCloud.png',format='png')
    plt.title('Word cloud')
    plt.clf()
    df['title_length'] = df['clean_title'].apply(len)
    print(df['title_length'].describe())
    word_counts = Counter(" ".join(df['clean_title']).split())
    print(word_counts.most_common(10))

def word_cloud_for_positive_and_negative_labels(df):
    """Generates the word cloud for true images and fake images.

    Args:
        df (_type_): _description_
    """
    positive_titles = df[df['2_way_label'] == 1]['clean_title']
    negative_titles = df[df['2_way_label'] == 0]['clean_title']

    positive_wordcloud = WordCloud(width=800, height=400,
                                   background_color='white').generate(' '.join(positive_titles))
    negative_wordcloud = WordCloud(width=800, height=400,
                                   background_color='white').generate(' '.join(negative_titles))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Class Word Cloud')

    plt.subplot(1, 2, 2)
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Class Word Cloud')

    plt.savefig(fname='results/plots/wordCloudPositiveAndNegative.png',format='png')


def box_plot_corelation_between_label_and_title_length(df):
    """Box plot title length and label.

    Args:
        df (data_frame): _description_
    """
    sns.boxplot(x='2_way_label', y='title_length', data=df)
    plt.title('Title Length by 2_way_label')
    plt.xlabel('2_way_label')
    plt.ylabel('Title Length')
    plt.savefig(fname='results/plots/boxPlot.png',format='png')


def eda(annotations_file):
    """performs exploratory data analysis on the dataset

    Args:
        annotations_file (_type_): the path to the annotation file
    """
    annotations = pd.read_csv(annotations_file,sep='\t',index_col='id')
    print(annotations.shape)
    print(annotations.info())

    label_distribution(annotations)
    title_analyzer(annotations)
    box_plot_corelation_between_label_and_title_length(annotations)
    word_cloud_for_positive_and_negative_labels(annotations)



if __name__ == "__main__":
    eda(annotations_file="./dataset/multimodal_only_samples/multimodal_train.tsv")
