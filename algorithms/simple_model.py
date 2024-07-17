from abc import ABC,abstractmethod
import os
from torchvision import transforms

import numpy as np


class SimpleModel():
    def __init__(self, train_features, train_labels, test_features, test_labels,model=None):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.model = model
        self.predictions = None
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
    
    @abstractmethod
    def train_and_test(self):
        """Trains and tests a linear regression model.
        Args:
            train_features (_type_): _description_
            train_labels (_type_): _description_
            test_features (_type_): _description_
            test_labels (_type_): _description_
        """
        pass 
    @abstractmethod
    def evaluate_results(self):
        pass
    
    def save_correct_incorrect_predictions(self, dataset, output_dir='results/predictions'):
        self.predictions = self.model.predict(self.test_features)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        correct_true_indices = np.where((self.predictions == self.test_labels) & (self.predictions == 1))[0]
        incorrect_true_indices = np.where((self.predictions != self.test_labels) & (self.predictions == 1))[0]
        correct_false_indices = np.where((self.predictions == self.test_labels) & (self.predictions == 0))[0]
        incorrect_false_indices = np.where((self.predictions != self.test_labels) & (self.predictions == 0))[0]


        def save_images(indices, subdir):
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
            
            for idx in indices:
                img, _ = dataset[idx]
                label = self.test_labels[idx]
                pred = self.predictions[idx]
                img.save(os.path.join(subdir_path, f'{idx}_pred{pred}_actual{label}.png'))

        save_images(correct_true_indices, 'correct_true')
        save_images(incorrect_true_indices, 'incorrect_true')
        save_images(correct_false_indices, 'correct_false')
        save_images(incorrect_false_indices, 'incorrect_false')
        print(f'Saved correct and incorrect predictions to {output_dir}')