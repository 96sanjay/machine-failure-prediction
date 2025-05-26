
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualizer:
    def __init__(self, config):
        self.config = config
        # plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_theme()
        sns.set_palette("husl")
        
    def plot_data_overview(self, df):
        """Create comprehensive data overview plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        sns.countplot(data=df, x='fail', ax=axes[0,0], palette='Set1')
        axes[0,0].set_title('Target Variable Distribution')
        axes[0,0].set_xlabel('Machine Failure')
        axes[0,0].set_ylabel('Count')
        
        # Missing values heatmap
        sns.heatmap(df.isnull(), cbar=True, ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title('Missing Values Heatmap')
        
        # Correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                            ax=axes[1,0], cmap='coolwarm', center=0)
        axes[1,0].set_title('Feature Correlation Matrix')
        
        # Data distribution
        df[numeric_cols].hist(bins=20, ax=axes[1,1], alpha=0.7)
        axes[1,1].set_title('Feature Distributions')
        
        plt.tight_layout()
        
        if self.config.output_config['save_plots']:
            plt.savefig("results/plots/data_overview.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_distributions(self, df):
        """Plot feature distributions by target class"""
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != 'fail']
        
        n_features = len(numeric_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(numeric_features):
            if i < len(axes):
                sns.boxplot(data=df, x='fail', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} Distribution by Target')
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        if self.config.output_config['save_plots']:
            plt.savefig("results/plots/feature_distributions.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, df, model_results):
        """Create interactive dashboard using Plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Distribution', 'Feature Correlations', 
                            'Model Performance', 'ROC Curves'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Target distribution
        target_counts = df['fail'].value_counts()
        fig.add_trace(
            go.Bar(x=['No Failure', 'Failure'], 
                     y=target_counts.values,
                     name='Target Distribution'),
            row=1, col=1
        )
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values,
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       colorscale='RdBu',
                       name='Correlations'),
            row=1, col=2
        )
        
        # Model performance comparison
        if model_results:
            models = list(model_results.keys())
            accuracies = [model_results[model].get('accuracy', 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name='Accuracy'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, showlegend=False, 
                          title_text="Machine Failure Prediction Dashboard")
        
        if self.config.output_config['save_plots']:
            fig.write_html("results/plots/interactive_dashboard.html")
        
        fig.show()
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            print(f"Model {model_name} doesn't have feature importance")
            return None
            
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(feature_names)), importance[indices])
        plt.xticks(range(len(feature_names)), 
                   [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if self.config.output_config['save_plots']:
            plt.savefig(f"results/plots/feature_importance_{model_name}.png", 
                                dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return importance, indices