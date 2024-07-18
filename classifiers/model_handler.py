from typing import Dict
from classifiers.simple_model import SimpleModel

class ModelHandler():
    """Handler that keeps track of the simple models used for classification. 
    """
    def __init__(self, models : Dict[str,SimpleModel]):
        """
        Args:
            models (Dict[str,SimpleModel]): the models that the handler handles. 
        """
        self.models = models
    def train_model(self, name):
        """Trains the specified model.

        Args:
            name (str): The name of the model specified 
        """
        self.models[name].train()
    def test_model(self, name : str):
        """Tests the specified model.

        Args:
            name (str): _description_
        """
        self.models[name].test()
    def evaluate_results(self, name : str):
        self.models[name].evaluate_results()

    def add_model(self,name :str,model : SimpleModel):
        self.models[name] = model
    
    def test_all(self):
        for model in self.models.values():
            model.test()
    
    def save_correct_incorrect_predictions_model(self,name:str,dataset,output_dir='results/predictions'):
        self.models[name].save_correct_incorrect_predictions(dataset,output_dir + name)