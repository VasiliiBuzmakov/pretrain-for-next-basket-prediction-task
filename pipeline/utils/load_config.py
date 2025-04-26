import os
import json
import torch
import itertools

    
class Config:
    
    def __init__(self, model_name = None, dataset_name = None, seed = None) -> None:
        abs_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(abs_path) as file:
            config = json.load(file)
        abs_path_models = os.path.join(os.path.dirname(__file__), "models_configs.json")
        with open(abs_path_models) as file:
            models_config = json.load(file)
        if model_name is None:
            model_name = config['model_name']
        if dataset_name is None:
            dataset_name = config['dataset_name']

        # model_settings = models_config[model_name[0]][dataset_name]
        
        for key in config:
            setattr(self, key, config[key])
        
        # for key in model_settings:
        #     setattr(self, key, model_settings[key])
        
        if seed:
            self.seed = seed

    def modify_config(self, model_name = None, dataset_name = None, **kwargs) :
        abs_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(abs_path) as file:
            config = json.load(file)
        abs_path_models = os.path.join(os.path.dirname(__file__), "models_configs.json")
        with open(abs_path_models) as file:
            models_config = json.load(file)
        if model_name is None:
            model_name = config['model_name']
        else:
            config['model_name'] = model_name

        if dataset_name is None:
            dataset_name = config['dataset_name']
        else:
            config['dataset_name'] = dataset_name
        # print(models_config)
        model_settings = models_config[model_name][dataset_name]

        # attributes = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        # for attr in attributes:
        #     delattr(self, attr)

        for key in config:
            setattr(self, key, config[key])
        
        for key in model_settings:
            setattr(self, key, model_settings[key])
        
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        attributes = [f"'{key}': '{value}' \n" for key, value in self.__dict__.items()]
        return '{' + ', '.join(attributes) + '}'

config = Config()

