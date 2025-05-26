
import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('machine_failure_prediction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_object(obj, filepath):
    """Save object using pickle"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Object saved to {filepath}")
    except Exception as e:
        print(f"Error saving object: {e}")

def load_object(filepath):
    """Load object using pickle"""
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        print(f"Error loading object: {e}")
        return None

def save_json(data, filepath):
    """Save data as JSON"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        print(f"JSON saved to {filepath}")
    except Exception as e:
        print(f"Error saving JSON: {e}")

def load_json(filepath):
    """Load JSON data"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def create_timestamp():
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def get_model_summary(model):
    """Get model summary information"""
    summary = {
        'model_type': type(model).__name__,
        'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A'
    }
    return summary

def calculate_memory_usage(df):
    """Calculate memory usage of dataframe"""
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    return f"{memory_usage:.2f} MB"

def print_system_info():
    """Print system information"""
    import platform
    import psutil
    
    print("System Information:")
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")