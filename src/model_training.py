
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pickle
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = self._initialize_models()
        self.hyperparameters = self._get_hyperparameters()
        self.best_models = {}
        self.model_results = {}
        
    def _initialize_models(self):
        """Initialize all models"""
        return {
            'SVC': SVC(probability=True),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'XGBClassifier': XGBClassifier(eval_metric='logloss')
        }
    
    def _get_hyperparameters(self):
        """Define hyperparameter grids for each model"""
        return {
            "SVC": {
                'C': [1.0, 5.0, 10.0, 15.0, 30.0, 50.0],
                'kernel': ['rbf'],
                'gamma': [0.1, 0.2, 0.5, 0.8, 1.0],
            },
            "DecisionTreeClassifier": {
                'criterion': ['gini'],
                'max_depth': [3, 5, 10, 15, 20],
                'min_samples_split': [2, 4, 6, 8, 10],
                'min_samples_leaf': [1, 5, 10],
                'splitter': ['best', 'random'],
            },
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 120, 150, 200],
                'criterion': ['gini'],
                "max_depth": [2, 3, 5, 8, 10, 20],
                'min_samples_split': [2, 4, 6, 7, 8, 9, 10],
            },
            "AdaBoostClassifier": {
                'n_estimators': [100, 200, 230, 250],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'algorithm': ['SAMME'],
            },
            "GradientBoostingClassifier": {
                'learning_rate': [0.01, 0.1, 0.2, 0.5, 0.8, 1.0],
                'n_estimators': [100, 200, 230, 250, 300],
                'max_depth': [3, 2, 5, 8, 10],
                'min_samples_split': [2, 3, 4, 8, 10, 12],
            },
            "XGBClassifier": {
                'learning_rate': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                'n_estimators': [120, 140, 180, 200],
                'max_depth': [1, 2, 3, 5, 8, 10, 15, 20],
                'gamma': [0.1, 0.2, 0.3, 0.5, 0.8],
                'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],
            }
        }
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models with default parameters"""
        print("Training baseline models...")
        baseline_results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Store results
            baseline_results[name] = {
                'model': model,
                'train_predictions': train_pred,
                'test_predictions': test_pred
            }
            
        return baseline_results
    
    def optimize_hyperparameters(self, X_train, X_test, y_train, y_test):
        """Optimize hyperparameters using GridSearchCV"""
        print("Starting hyperparameter optimization...")
        
        cv_folds = self.config.hyperparameters_config['cv_folds']
        scoring = self.config.hyperparameters_config['scoring']
        n_jobs = self.config.hyperparameters_config['n_jobs']
        
        for name in self.models.keys():
            print(f"\nOptimizing {name}...")
            
            model = self.models[name]
            params = self.hyperparameters[name]
            
            # Grid Search
            grid_search = GridSearchCV(
                model, 
                params, 
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best model and results
            self.best_models[name] = grid_search.best_estimator_
            
            # Test predictions
            test_pred = grid_search.best_estimator_.predict(X_test)
            test_proba = None
            
            try:
                test_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
            except:
                pass
            
            self.model_results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_predictions': test_pred,
                'test_probabilities': test_proba,
                'grid_search': grid_search
            }
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        return self.best_models, self.model_results
    
    def save_models(self):
        """Save trained models"""
        if not self.config.output_config['save_models']:
            return
            
        models_dir = self.config.output_config['models_dir']
        os.makedirs(models_dir, exist_ok=True)
        
        for name, model in self.best_models.items():
            filename = os.path.join(models_dir, f"{name}_best.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filename}")
    
    def load_model(self, model_name, models_dir=None):
        """Load a saved model"""
        if models_dir is None:
            models_dir = self.config.output_config['models_dir']
            
        filename = os.path.join(models_dir, f"{model_name}_best.pkl")
        
        try:
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded {model_name} from {filename}")
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None