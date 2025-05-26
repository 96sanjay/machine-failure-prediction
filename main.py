
#!/usr/bin/env python3
"""
Main script for Machine Failure Prediction System
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from config import Config
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from visualization import DataVisualizer
from utils import setup_logging, print_system_info

def main():
    """Main execution function"""
    print("=" * 60)
    print("MACHINE FAILURE PREDICTION SYSTEM")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Machine Failure Prediction System")
    
    # Print system information
    print_system_info()
    
    try:
        # Load configuration
        config = Config()
        config.create_directories()
        
        # Initialize components
        preprocessor = DataPreprocessor(config)
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)
        visualizer = DataVisualizer(config)
        
        # Data preprocessing
        print("\n" + "=" * 40)
        print("DATA PREPROCESSING")
        print("=" * 40)
        
        preprocessed_data = preprocessor.preprocess_pipeline()
        
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']
        y_train = preprocessed_data['y_train']
        y_test = preprocessed_data['y_test']
        
        # Visualize data
        print("\n" + "=" * 40)
        print("DATA VISUALIZATION")
        print("=" * 40)
        
        if config.output_config['save_plots']:
            visualizer.plot_data_overview(preprocessor.df)
            visualizer.plot_feature_distributions(preprocessor.df)
        
        # Model training
        print("\n" + "=" * 40)
        print("MODEL TRAINING & OPTIMIZATION")
        print("=" * 40)
        
        # Train baseline models
        baseline_results = trainer.train_baseline_models(X_train, X_test, y_train, y_test)
        
        # Optimize hyperparameters
        best_models, model_results = trainer.optimize_hyperparameters(
            X_train, X_test, y_train, y_test
        )
        
        # Save models
        trainer.save_models()
        
        # Model evaluation
        print("\n" + "=" * 40)
        print("MODEL EVALUATION")
        print("=" * 40)
        
        # Evaluate models
        comparison_df = evaluator.evaluate_models(model_results, y_test)
        
        print("\nModel Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        # Get best model
        best_model_name, best_score = evaluator.get_best_model(comparison_df)
        
        # Generate detailed evaluation for best model
        evaluator.generate_classification_report(best_model_name, model_results, y_test)
        
        # Create visualizations
        if config.output_config['save_plots']:
            evaluator.plot_confusion_matrix(best_model_name, model_results, y_test)
            evaluator.plot_roc_curve(best_model_name, model_results, y_test)
            evaluator.plot_model_comparison(comparison_df)
            
            # Feature importance for tree-based models
            if 'Forest' in best_model_name or 'Tree' in best_model_name or 'GradientBoosting' in best_model_name or 'XGB' in best_model_name:
                feature_names = preprocessor.df.drop('fail', axis=1).columns
                visualizer.plot_feature_importance(
                    best_models[best_model_name], 
                    feature_names, 
                    best_model_name
                )
        
        # Save results
        evaluator.save_results(comparison_df, best_model_name)
        
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETED SUCCESSFULLY!")
        print(f"Best Model: {best_model_name}")
        print(f"Best ROC-AUC Score: {best_score:.4f}")
        print("Check 'results/' directory for detailed outputs")
        print("=" * 60)
        
        logger.info("Machine Failure Prediction System completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)