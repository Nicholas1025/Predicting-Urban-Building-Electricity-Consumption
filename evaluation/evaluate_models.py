"""
Model Evaluation Script for Building Energy Consumption Prediction
Evaluates multiple regression models and creates comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_true_values():
    """
    Load the true test values
    
    Returns:
        pd.Series: True test values
    """
    try:
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        print(f"True values loaded: {len(y_test)} samples")
        return y_test
    except FileNotFoundError:
        print("Error: y_test.csv not found in outputs directory")
        print("Please run clean_data.py first to generate test data")
        return None
    except Exception as e:
        print(f"Error loading true values: {e}")
        return None


def load_model_predictions():
    """
    Load predictions from all available models
    
    Returns:
        dict: Dictionary with model names as keys and predictions as values
    """
    predictions = {}
    
    # Find all prediction files
    prediction_files = glob.glob("outputs/predictions_*.csv")
    
    if not prediction_files:
        print("No prediction files found in outputs directory")
        print("Please run model training scripts first")
        return predictions
    
    print(f"Found {len(prediction_files)} prediction files:")
    
    for file_path in prediction_files:
        try:
            # Extract model name from filename
            filename = os.path.basename(file_path)
            model_name = filename.replace("predictions_", "").replace(".csv", "").upper()
            
            # Load predictions
            pred_df = pd.read_csv(file_path)
            predictions[model_name] = pred_df.iloc[:, 0].values  # First column
            
            print(f"  ✓ {model_name}: {len(predictions[model_name])} predictions")
            
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {e}")
    
    return predictions


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics


def evaluate_all_models(y_true, predictions_dict):
    """
    Evaluate all models and return results
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for each model
    """
    results = []
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    for model_name, y_pred in predictions_dict.items():
        # Ensure predictions have the same length as true values
        if len(y_pred) != len(y_true):
            print(f"Warning: {model_name} predictions length mismatch")
            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true_subset = y_true.iloc[:min_len]
        else:
            y_true_subset = y_true
        
        # Calculate metrics
        metrics = calculate_metrics(y_true_subset, y_pred)
        
        # Add to results
        result_row = {'Model': model_name}
        result_row.update(metrics)
        results.append(result_row)
        
        # Print results
        print(f"\n{model_name} Model:")
        print(f"  MAE:  {metrics['MAE']:.2f}")
        print(f"  RMSE: {metrics['RMSE']:.2f}")
        print(f"  R²:   {metrics['R2']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R2', ascending=False)  # Sort by R² (best first)
    
    print(f"\n" + "="*60)
    print("RANKING (by R² Score):")
    print("="*60)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i}. {row['Model']:<15} R² = {row['R2']:.4f}")
    
    return results_df


def plot_predicted_vs_actual(y_true, predictions_dict, save_dir="outputs/charts"):
    """
    Create predicted vs actual scatter plots for each model
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        save_dir (str): Directory to save plots
    """
    print(f"\nCreating predicted vs actual plots...")
    
    n_models = len(predictions_dict)
    if n_models == 0:
        return
    
    # Determine subplot layout
    if n_models == 1:
        nrows, ncols = 1, 1
    elif n_models == 2:
        nrows, ncols = 1, 2
    elif n_models <= 4:
        nrows, ncols = 2, 2
    elif n_models <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Calculate global min and max for consistent axis scaling
    all_values = list(y_true)
    for pred in predictions_dict.values():
        all_values.extend(pred)
    
    global_min = min(all_values)
    global_max = max(all_values)
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        # Ensure same length
        if len(y_pred) != len(y_true):
            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true_subset = y_true.iloc[:min_len]
        else:
            y_true_subset = y_true
        
        # Calculate R² for the plot
        r2 = r2_score(y_true_subset, y_pred)
        
        # Create scatter plot
        ax.scatter(y_true_subset, y_pred, alpha=0.6, s=20)
        
        # Add perfect prediction line
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', lw=2, label='Perfect Prediction')
        
        # Set labels and title
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name} Model\nR² = {r2:.4f}')
        
        # Set consistent axis limits
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, "predicted_vs_actual_all_models.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predicted vs Actual plots saved to: {save_path}")


def plot_model_comparison(results_df, save_dir="outputs/charts"):
    """
    Create comparison bar charts for model performance
    
    Args:
        results_df (pd.DataFrame): Results dataframe with metrics
        save_dir (str): Directory to save plots
    """
    print(f"\nCreating model comparison charts...")
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MAE', 'RMSE', 'R2']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        
        # Sort by current metric (ascending for MAE/RMSE, descending for R²)
        if metric in ['MAE', 'RMSE']:
            sorted_df = results_df.sort_values(metric, ascending=True)
            title_suffix = "(Lower is Better)"
        else:
            sorted_df = results_df.sort_values(metric, ascending=False)
            title_suffix = "(Higher is Better)"
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric], color=color, edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(sorted_df[metric]) - min(sorted_df[metric])) * 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Customize plot
        ax.set_title(f'{metric} Comparison\n{title_suffix}', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xlabel('Models')
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add some padding to y-axis
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.1)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, "model_comparison_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison charts saved to: {save_path}")


def plot_residuals_analysis(y_true, predictions_dict, save_dir="outputs/charts"):
    """
    Create residual plots for model analysis
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        save_dir (str): Directory to save plots
    """
    print(f"\nCreating residual analysis plots...")

    n_models = min(len(predictions_dict), 3)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))

    # Handle axes shape when n_models == 1
    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        if idx >= 3:
            break

        if len(y_pred) != len(y_true):
            min_len = min(len(y_pred), len(y_true))
            y_pred = y_pred[:min_len]
            y_true_subset = y_true.iloc[:min_len]
        else:
            y_true_subset = y_true

        residuals = y_true_subset - y_pred

        # Plot residuals vs predicted
        ax1 = axes[0, idx]
        ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'{model_name} - Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)

        # Plot histogram of residuals
        ax2 = axes[1, idx]
        ax2.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='navy')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{model_name} - Residuals Distribution')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "residuals_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residual analysis plots saved to: {save_path}")


def save_results_summary(results_df, save_dir="outputs"):
    """
    Save evaluation results to CSV file
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save_dir (str): Directory to save results
    """
    save_path = os.path.join(save_dir, "model_evaluation_results.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nEvaluation results saved to: {save_path}")


def main():
    """
    Main evaluation pipeline
    """
    print("="*60)
    print("MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    
    # Load true values
    y_true = load_true_values()
    if y_true is None:
        return
    
    # Load model predictions
    predictions_dict = load_model_predictions()
    if not predictions_dict:
        return
    
    # Evaluate all models
    results_df = evaluate_all_models(y_true, predictions_dict)
    
    # Create visualizations
    plot_predicted_vs_actual(y_true, predictions_dict)
    plot_model_comparison(results_df)
    plot_residuals_analysis(y_true, predictions_dict)
    
    # Save results
    save_results_summary(results_df)
    
    print(f"\n" + "="*60)
    print("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files created:")
    print("  - outputs/model_evaluation_results.csv")
    print("  - outputs/charts/predicted_vs_actual_all_models.png")
    print("  - outputs/charts/model_comparison_metrics.png")
    print("  - outputs/charts/residuals_analysis.png")
    
    # Print final summary
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    best_model = results_df.iloc[0]
    print(f"🏆 Best Model: {best_model['Model']}")
    print(f"   R² Score: {best_model['R2']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.2f}")
    print(f"   MAE: {best_model['MAE']:.2f}")
    
    if len(results_df) > 1:
        print(f"\n📊 Model Performance Ranking:")
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"   {i}. {row['Model']:<15} (R² = {row['R2']:.4f})")


if __name__ == "__main__":
    main()