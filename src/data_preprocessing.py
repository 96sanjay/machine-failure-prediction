import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer()
        
    def load_data(self, file_path=None):
        """Load dataset from CSV file"""
        if file_path is None:
            file_path = self.config.data_config['file_path']
            
        try:
            self.df = pd.read_csv(file_path)
            print(f"Data loaded successfully: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Perform initial data exploration"""
        print("Dataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        print(f"Target distribution:\n{self.df['fail'].value_counts()}")
        
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'target_distribution': self.df['fail'].value_counts().to_dict()
        }
    
    def handle_missing_values(self):
        """Handle missing values based on configuration"""
        missing_counts = self.df.isnull().sum()
        
        if missing_counts.sum() == 0:
            print("No missing values found.")
            return self.df
            
        method = self.config.preprocessing_config['handle_missing']
        
        if method == 'drop':
            self.df = self.df.dropna()
            print(f"Missing values dropped. New shape: {self.df.shape}")
        elif method == 'impute':
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_columns] = self.imputer.fit_transform(self.df[numeric_columns])
            print("Missing values imputed.")
            
        return self.df
    
    def prepare_features(self):
        """Prepare features and target variables"""
        # Separate features and target
        X = self.df.drop('fail', axis=1)
        y = self.df['fail']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into training and testing sets"""
        test_size = self.config.data_config['test_size']
        random_state = self.config.data_config['random_state']
        
        # Note: Using train_size=0.2 means 20% for training, 80% for testing
        # This seems unusual - typically we use 80% for training
        # I'll correct this to use test_size parameter properly
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,  # 20% for testing, 80% for training
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Testing set: X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        scaling_method = self.config.preprocessing_config['scaling_method']
        
        if scaling_method == 'standard':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            print("Features scaled using StandardScaler")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            print("No scaling applied")
            
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, file_path=None):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        self.load_data(file_path)
        
        # Explore data
        data_info = self.explore_data()
        
        # Handle missing values
        self.handle_missing_values()
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("Preprocessing pipeline completed successfully!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': self.scaler,
            'data_info': data_info
        }