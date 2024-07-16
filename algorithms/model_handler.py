from typing import Dict
from algorithms.simple_model import SimpleModel

class ModelHandler():
    def __init__(self, models : Dict[str,SimpleModel]):
        """
        Args:
            models (Dict[str,SimpleModel]): _description_
        """
        self.models = models
    def train_model(self, name):
        self.models[name].train()
    def test_model(self, name):
        self.models[name].test()
    def evaluate_results(self, name):
        self.models[name].evaluate_results()

    def add_model(self,name,model):
        self.models[name] = model
    
    def test_all(self):
        for model in self.models.values():
            model.test()