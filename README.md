# Experiments with pre-train for Next Basket Prediction task

## Adding new models

1) create a folder with the name of your model in the models folder.
2) This folder must contain three files:
- model_creation.py with the create_model function
- prepare_dataloader.py with the function prepare_dataloader
- train_eval_routine.py with functions train_epoch, eval_epoch, evaluate

3) In models_configs.json add a dictionary with the configuration and the required dataset for your model.
In Main.py and evaluate.py the configuration file is an object called opt. 
It is passed to all the functions in the 3 files described above. To access any characteristic of the "config file", you can do so by referring to an attribute with the same name, e.g. opt.batch_size

When loading a config file, config.json is loaded first, then the dictionary from models_configs.json corresponding to the model and dataset, if there are configuration parameters with the same name in config.json and models_configs.json, then the ones specified in models_configs.json will be used. So, for example, you can store a specific learning rate for a specific model on a specific dataset.

## Adding datasets

The data is stored in the tcmbn_data folder. You can see an example of data preprocessing in preprocess.ipynb
