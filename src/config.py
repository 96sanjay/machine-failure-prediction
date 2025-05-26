import yaml
import os
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @property
    def data_config(self):
        return self.config['data']
    
    @property
    def preprocessing_config(self):
        return self.config['preprocessing']
    
    @property
    def models_config(self):
        return self.config['models']
    
    @property
    def hyperparameters_config(self):
        return self.config['hyperparameters']
    
    @property
    def output_config(self):
        return self.config['output']
    
    def create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            'data/raw',
            'data/processed',  
            'models/trained_models',
            'results/plots',
            'results/reports',
            'notebooks',
            'tests'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)