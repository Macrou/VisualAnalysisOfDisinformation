from abc import ABC,abstractmethod


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
    