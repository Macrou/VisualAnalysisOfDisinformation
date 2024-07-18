# Visual Analysis of disinformation
## Geting started 
### Download datasets
Download [Kai Nakamura, Sharon Levy, and William Yang Wang. 2020. r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://github.com/entitize/Fakeddit/blob/master/README.md) and [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) datasets .
### Download the requirements.
Use `pip install -r requirements.txt` to install or the dependencies necessary for the project
### Runing the pipeline
To run the clip pipeline run `python clip_detection_model.py` with the appropriate flags mentioned in options_clip.py. 
For the Resnet pipeline run `python resnet_detection.py` with the flags mentioned in resnet_detection.py
For the Multimodal pipeline run `python multimodal_clip_detection_model.py` with the flags mentioned. 
## Known issues
For the multimodal pipeline there is an issue when the context of the text that needs to be tokenized is bigger than 70 that needs to be solved. 
