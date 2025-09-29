"""
Visualization module for Energy Consumption Predictor.
Creates plots for historical consumption, forecasts, actuals vs predictions, and feature importance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnergyVisualizationSuite:
    """
    Comprehensive visualization suite for energy consumption analysis and prediction results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), save_path: str = 'visualizations'):
        """
        Initialize the visualization suite.
        
        Args:
            figsize: Default figure size for matplotlib plots
            save_path: Directory to save plots
        """
        self.figsize = figsize
        self.save_path = save_path
        
        # Create save directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Color palette
        self.colors = {
            'actual': '#2E86AB',
            'predicted': '#F24236',
            'baseline': '#F6AE2D',
            'confidence': '#A8DADC',
            'feature_importance': '#457B9D'
        }
    
    def plot_time_series(self, df: pd.DataFrame, 
                        columns: List[str] = None,
                        title: str = "Energy Consumption Time Series",
                        ylabel: str = "Energy Consumption (kWh)",
                        save_name: str = "time_series.png",
                        interactive: bool = False) -> None:
        """
        Plot time series data.
        
        Args:
            df: DataFrame with datetime index and consumption data
            columns: Columns to plot (default: ['kwh'])
            title: Plot title
            ylabel: Y-axis label
            save_name: Filename to save the plot
            interactive: Use plotly for interactive plot
        """
        if columns is None:
            columns = ['kwh'] if 'kwh' in df.columns else [df.columns[0]]
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            for col in columns:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=ylabel,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Save as HTML
            save_path = f"{self.save_path}/{save_name.replace('.png', '.html')}"
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
            
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for col in columns:
                if col in df.columns:
                    ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), linewidth=2)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            save_path = f"{self.save_path}/{save_name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()
    
    def plot_predictions_vs_actual(self, y_true: pd.Series, 
                                  predictions: Dict[str, np.ndarray],
                                  title: str = "Predictions vs Actual",
                                  save_name: str = "predictions_vs_actual.png",
                                  interactive: bool = False) -> None:
        """
        Plot predictions against actual values for multiple models.
        
        Args:
            y_true: Actual values
            predictions: Dictionary with model names as keys and predictions as values
            title: Plot title
            save_name: Filename to save the plot
            interactive: Use plotly for interactive plot
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Add actual values
            fig.add_trace(go.Scatter(
                x=y_true.index,
                y=y_true.values,
                mode='lines',
                name='Actual',
                line=dict(color=self.colors['actual'], width=3)
            ))
            
            # Add predictions
            for i, (model_name, pred_values) in enumerate(predictions.items()):
                if len(pred_values) == len(y_true):
                    fig.add_trace(go.Scatter(
                        x=y_true.index,
                        y=pred_values,
                        mode='lines',
                        name=model_name.replace('_', ' ').title(),
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Energy Consumption (kWh)",
                hovermode='x unified',
                template='plotly_white'
            )
            
            save_path = f"{self.save_path}/{save_name.replace('.png', '.html')}"
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
            
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot actual values
            ax.plot(y_true.index, y_true.values, 
                   label='Actual', color=self.colors['actual'], linewidth=3, alpha=0.8)
            
            # Plot predictions
            colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
            for i, (model_name, pred_values) in enumerate(predictions.items()):
                if len(pred_values) == len(y_true):
                    ax.plot(y_true.index, pred_values,
                           label=model_name.replace('_', ' ').title(),
                           color=colors[i], linewidth=2, alpha=0.7)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Energy Consumption (kWh)", fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = f"{self.save_path}/{save_name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                               top_n: int = 20,
                               title: str = "Feature Importance",
                               save_name: str = "feature_importance.png",
                               interactive: bool = False) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_dict: Dictionary with feature names and importance values
            top_n: Number of top features to display
            title: Plot title
            save_name: Filename to save the plot
            interactive: Use plotly for interactive plot
        """
        if not importance_dict:
            print("No feature importance data available")
            return
        
        # Sort features by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color=self.colors['feature_importance']
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance",
                yaxis_title="Features",
                template='plotly_white',
                height=max(400, len(features) * 20)
            )
            
            save_path = f"{self.save_path}/{save_name.replace('.png', '.html')}"
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
            
        else:
            # Matplotlib version
            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
            
            bars = ax.barh(range(len(features)), importances, 
                          color=self.colors['feature_importance'], alpha=0.7)
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f.replace('_', ' ') for f in features])
            ax.set_xlabel("Importance", fontsize=12)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01 * max(importances), bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            save_path = f"{self.save_path}/{save_name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()
    
    def plot_model_comparison(self, metrics_dict: Dict[str, Dict[str, float]],
                             metrics_to_plot: List[str] = ['mae', 'rmse', 'mape'],
                             title: str = "Model Performance Comparison",
                             save_name: str = "model_comparison.png",
                             interactive: bool = False) -> None:
        """
        Compare model performance across multiple metrics.
        
        Args:
            metrics_dict: Dictionary with model names and their metrics
            metrics_to_plot: List of metrics to include in the plot
            title: Plot title
            save_name: Filename to save the plot
            interactive: Use plotly for interactive plot
        """
        if not metrics_dict:
            print("No metrics data available")
            return
        
        # Prepare data
        models = list(metrics_dict.keys())
        
        if interactive and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=1, cols=len(metrics_to_plot),
                subplot_titles=metrics_to_plot,
                shared_yaxis=True
            )
            
            for i, metric in enumerate(metrics_to_plot):
                values = [metrics_dict[model].get(metric, 0) for model in models]
                
                fig.add_trace(
                    go.Bar(x=models, y=values, name=metric.upper()),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title=title,
                template='plotly_white',
                showlegend=False
            )
            
            save_path = f"{self.save_path}/{save_name.replace('.png', '.html')}"
            fig.write_html(save_path)
            print(f"Interactive plot saved to {save_path}")
            
        else:
            # Matplotlib version
            fig, axes = plt.subplots(1, len(metrics_to_plot), 
                                   figsize=(5 * len(metrics_to_plot), 6))
            
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics_to_plot):
                values = [metrics_dict[model].get(metric, 0) for model in models]
                
                bars = axes[i].bar(models, values, alpha=0.7, 
                                  color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                
                axes[i].set_title(metric.upper(), fontsize=14, fontweight='bold')
                axes[i].set_ylabel(metric.upper(), fontsize=12)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            save_path = f"{self.save_path}/{save_name}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()
    
    def plot_residuals(self, y_true: pd.Series, y_pred: np.ndarray,
                      model_name: str = "Model",
                      title: str = None,
                      save_name: str = "residuals.png") -> None:
        """
        Plot residual analysis.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            title: Plot title
            save_name: Filename to save the plot
        """
        if title is None:
            title = f"Residual Analysis - {model_name}"
        
        # Calculate residuals
        residuals = y_true.values - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=self.colors['predicted'])
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['predicted'], edgecolor='black')
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Distribution of Residuals")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals over time
        axes[1, 1].plot(y_true.index, residuals, alpha=0.7, color=self.colors['predicted'])
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].set_title("Residuals Over Time")
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.save_path}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual analysis saved to {save_path}")
        plt.close()
    
    def plot_seasonal_patterns(self, df: pd.DataFrame, 
                              value_col: str = 'kwh',
                              title: str = "Seasonal Patterns",
                              save_name: str = "seasonal_patterns.png") -> None:
        """
        Plot seasonal patterns in the data.
        
        Args:
            df: DataFrame with datetime index and consumption data
            value_col: Column containing the values to analyze
            title: Plot title
            save_name: Filename to save the plot
        """
        if value_col not in df.columns:
            print(f"Column {value_col} not found in DataFrame")
            return
        
        # Prepare data
        df_copy = df.copy()
        df_copy['hour'] = df_copy.index.hour
        df_copy['day_of_week'] = df_copy.index.day_name()
        df_copy['month'] = df_copy.index.month_name()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hourly pattern
        hourly_avg = df_copy.groupby('hour')[value_col].mean()
        axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', 
                       color=self.colors['actual'], linewidth=2)
        axes[0, 0].set_xlabel("Hour of Day")
        axes[0, 0].set_ylabel(f"Average {value_col}")
        axes[0, 0].set_title("Average Consumption by Hour")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Daily pattern
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_avg = df_copy.groupby('day_of_week')[value_col].mean().reindex(day_order)
        axes[0, 1].bar(daily_avg.index, daily_avg.values, 
                      color=self.colors['feature_importance'], alpha=0.7)
        axes[0, 1].set_xlabel("Day of Week")
        axes[0, 1].set_ylabel(f"Average {value_col}")
        axes[0, 1].set_title("Average Consumption by Day")
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Monthly pattern
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg = df_copy.groupby('month')[value_col].mean().reindex(month_order)
        axes[1, 0].bar(range(len(monthly_avg)), monthly_avg.values, 
                      color=self.colors['baseline'], alpha=0.7)
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel(f"Average {value_col}")
        axes[1, 0].set_title("Average Consumption by Month")
        axes[1, 0].set_xticks(range(len(monthly_avg)))
        axes[1, 0].set_xticklabels([m[:3] for m in monthly_avg.index], rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Heatmap: Hour vs Day of Week
        pivot_data = df_copy.pivot_table(values=value_col, index='hour', 
                                        columns='day_of_week', aggfunc='mean')
        pivot_data = pivot_data[day_order]  # Reorder columns
        
        im = axes[1, 1].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(len(day_order)))
        axes[1, 1].set_xticklabels([d[:3] for d in day_order])
        axes[1, 1].set_yticks(range(0, 24, 2))
        axes[1, 1].set_yticklabels(range(0, 24, 2))
        axes[1, 1].set_xlabel("Day of Week")
        axes[1, 1].set_ylabel("Hour of Day")
        axes[1, 1].set_title("Consumption Heatmap (Hour vs Day)")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.save_path}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Seasonal patterns plot saved to {save_path}")
        plt.close()
    
    def create_dashboard_summary(self, results_dict: Dict[str, Any],
                                save_name: str = "dashboard_summary.png") -> None:
        """
        Create a comprehensive dashboard summary plot.
        
        Args:
            results_dict: Dictionary containing all analysis results
            save_name: Filename to save the dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Time series plot (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'predictions' in results_dict and 'y_test' in results_dict['predictions']:
            y_test = results_dict['predictions']['y_test']
            ax1.plot(y_test.index, y_test.values, label='Actual', 
                    color=self.colors['actual'], linewidth=2)
            
            # Plot best model predictions
            for model_name, pred in results_dict['predictions'].items():
                if model_name != 'y_test' and isinstance(pred, np.ndarray):
                    ax1.plot(y_test.index, pred, label=model_name.title(), 
                            alpha=0.7, linewidth=1.5)
                    break  # Only plot one model to avoid clutter
        
        ax1.set_title("Energy Consumption Predictions", fontweight='bold')
        ax1.set_ylabel("kWh")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Model performance comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'metrics' in results_dict:
            models = list(results_dict['metrics'].keys())[:5]  # Top 5 models
            rmse_values = [results_dict['metrics'][m].get('rmse', 0) for m in models]
            
            bars = ax2.bar(models, rmse_values, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
            ax2.set_title("Model Performance (RMSE)", fontweight='bold')
            ax2.set_ylabel("RMSE")
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, rmse_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Feature importance (middle left, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'feature_importance' in results_dict:
            importance_dict = results_dict['feature_importance']
            if importance_dict:
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:10]
                features, importances = zip(*top_features)
                
                bars = ax3.barh(range(len(features)), importances, 
                               color=self.colors['feature_importance'], alpha=0.7)
                ax3.set_yticks(range(len(features)))
                ax3.set_yticklabels([f.replace('_', ' ')[:20] + '...' if len(f) > 20 else f.replace('_', ' ') 
                                    for f in features])
                ax3.set_title("Top 10 Feature Importance", fontweight='bold')
                ax3.set_xlabel("Importance")
        
        # 4. Residuals (middle right)
        ax4 = fig.add_subplot(gs[1, 2:])
        if 'predictions' in results_dict and 'y_test' in results_dict['predictions']:
            y_test = results_dict['predictions']['y_test']
            # Use first available prediction for residuals
            for model_name, pred in results_dict['predictions'].items():
                if model_name != 'y_test' and isinstance(pred, np.ndarray):
                    residuals = y_test.values - pred
                    ax4.scatter(pred, residuals, alpha=0.6, s=20)
                    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
                    ax4.set_title(f"Residuals - {model_name.title()}", fontweight='bold')
                    ax4.set_xlabel("Predicted Values")
                    ax4.set_ylabel("Residuals")
                    ax4.grid(True, alpha=0.3)
                    break
        
        # 5. Error distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'metrics' in results_dict:
            metrics_df = pd.DataFrame(results_dict['metrics']).T
            if 'mae' in metrics_df.columns:
                metrics_df['mae'].hist(bins=10, ax=ax5, alpha=0.7, 
                                      color=self.colors['baseline'], edgecolor='black')
                ax5.set_title("Distribution of MAE Across Models", fontweight='bold')
                ax5.set_xlabel("MAE")
                ax5.set_ylabel("Frequency")
                ax5.grid(axis='y', alpha=0.3)
        
        # 6. Summary statistics (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('off')  # Turn off axes for text
        
        # Add summary text
        summary_text = "Model Summary\n" + "="*20 + "\n"
        
        if 'metrics' in results_dict:
            best_model = min(results_dict['metrics'].items(), 
                           key=lambda x: x[1].get('rmse', float('inf')))
            
            summary_text += f"Best Model: {best_model[0].title()}\n"
            summary_text += f"RMSE: {best_model[1].get('rmse', 'N/A'):.3f}\n"
            summary_text += f"MAE: {best_model[1].get('mae', 'N/A'):.3f}\n"
            summary_text += f"MAPE: {best_model[1].get('mape', 'N/A'):.2f}%\n\n"
        
        if 'data_info' in results_dict:
            data_info = results_dict['data_info']
            summary_text += f"Data Points: {data_info.get('total_records', 'N/A')}\n"
            summary_text += f"Date Range: {data_info.get('duration_days', 'N/A')} days\n"
        
        ax6.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.suptitle("Energy Consumption Prediction Dashboard", 
                    fontsize=18, fontweight='bold', y=0.95)
        
        save_path = f"{self.save_path}/{save_name}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard summary saved to {save_path}")
        plt.close()

def main():
    """Example usage of the visualization suite."""
    # This would typically be called from the main training pipeline
    print("Visualization Suite - Example Usage")
    print("This module is designed to be imported and used with actual data.")
    print("See the training pipeline and notebooks for complete examples.")

if __name__ == "__main__":
    main()