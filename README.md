# LLMPlayGround
Practice LLM 

Train_v5.py
is to understand machine learning workflow from working with data, creating models, optimizing model parameters, and saving the trained models. This example is from PyTorch tutorial. Concepts I have learnt:
Datasets and DataLoaders
Tensors
NN Model may have several layers
Transformers
Parameter Optimization

Train_v3.py
succesfully ran inference on llama3.2-1b-Instruct model
The tokenizer encodes the input text into a format(Tensors) the model layers will understand. The dataset is processed through several
layers in the model. The output is then decoded by the tokenizer to a human readable format.

Custommodel.py and Custommodel_v2.py
Attempted to create a model from the params.config json file that comes when you download meta-llama weights. Used this model to run inference in train_v3.py
Learnt that a model may be composed of several layers through which the dataset(text or image) is passed for processing.
