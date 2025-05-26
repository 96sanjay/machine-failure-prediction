
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
            
        return metrics
    
    def evaluate_models(self, model_results, y_test):
        """Evaluate all models and create comparison"""
        print("Evaluating models...")
        
        results_list = []
        
        for name, results in model_results.items():
            y_pred = results['test_predictions']
            y_proba = results.get('test_probabilities')
            
            metrics = self.calculate_metrics(y_test, y_pred, y_proba)
            
            result_row = {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'ROC_AUC': metrics['roc_auc'],
                'F1_Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            }
            
            results_list.append(result_row)
            self.evaluation_results[name] = metrics
            
            print(f"{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.sort_values('ROC_AUC', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, comparison_df, metric='ROC_AUC'):
        """Get the best performing model based on specified metric"""
        best_model_name = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0][metric]
        
        print(f"Best model: {best_model_name}")
        print(f"Best {metric}: {best_score:.4f}")
        
        return best_model_name, best_score
    
    def generate_classification_report(self, model_name, model_results, y_test):
        """Generate detailed classification report"""
        y_pred = model_results[model_name]['test_predictions']
        
        print(f"Classification Report for {model_name}:")
        print("-" * 50)
        report = classification_report(y_test, y_pred)
        print(report)
        
        return report
    
    def plot_confusion_matrix(self, model_name, model_results, y_test):
        """Plot confusion matrix"""
        y_pred = model_results[model_name]['test_predictions']
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No Failure(0)', 'Failure(1)'],
                           yticklabels=['No Failure(0)', 'Failure(1)'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.config.output_config['save_plots']:
            plt.savefig(f"results/plots/confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, model_name, model_results, y_test):
        """Plot ROC curve"""
        y_proba = model_results[model_name].get('test_probabilities')
        
        if y_proba is None:
            print(f"No probabilities available for {model_name}")
            return None
            
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        if self.config.output_config['save_plots']:
            plt.savefig(f"results/plots/roc_curve_{model_name}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_model_comparison(self, comparison_df):
        """Plot model comparison chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics = ['Accuracy', 'ROC_AUC', 'F1_Score', 'Precision']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[idx])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.config.output_config['save_plots']:
            plt.savefig("results/plots/model_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, comparison_df, best_model_name):
        """Save evaluation results to file"""
        results_dir = "results/reports/"
        import os
        os.makedirs(results_dir, exist_ok=True)
        
        # Save comparison DataFrame
        comparison_df.to_csv(f"{results_dir}/model_comparison.csv", index=False)
        
        # Save detailed results
        with open(f"{results_dir}/evaluation_summary.txt", 'w') as f:
            f.write("Model Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n\n")
            f.write("Model Comparison:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            for model_name, metrics in self.evaluation_results.items():
                f.write(f"{model_name} Detailed Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
        
        print(f"Results saved to {results_dir}")