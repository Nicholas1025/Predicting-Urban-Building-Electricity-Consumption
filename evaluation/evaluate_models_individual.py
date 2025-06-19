"""
Model Evaluation Script for Individual Dataset Analysis
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


def get_output_paths(dataset_name):
    """Get output paths for specific dataset"""
    if not dataset_name:
        base_dir = "outputs"
    else:
        base_dir = f"outputs/{dataset_name}"
    
    return {
        'data_dir': base_dir,
        'charts_dir': f"{base_dir}/charts",
        'predicted_vs_actual_path': f"{base_dir}/charts/predicted_vs_actual_all_models.png",
        'model_comparison_path': f"{base_dir}/charts/model_comparison_metrics.png",
        'residuals_path': f"{base_dir}/charts/residuals_analysis.png",
        'results_path': f"{base_dir}/model_evaluation_results.csv"
    }


def load_true_values(dataset_name=None):
    """
    Load the true test values for specific dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        pd.Series: True test values
    """
    paths = get_output_paths(dataset_name)
    data_dir = paths['data_dir']
    
    try:
        y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
        print(f"True values loaded: {len(y_test)} samples from {dataset_name}")
        return y_test
    except FileNotFoundError:
        print(f"Error: y_test.csv not found in {data_dir}")
        print("Please run data preprocessing first to generate test data")
        return None
    except Exception as e:
        print(f"Error loading true values: {e}")
        return None


def load_model_predictions(dataset_name=None):
    """
    Load predictions from all available models for specific dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Dictionary with model names as keys and predictions as values
    """
    paths = get_output_paths(dataset_name)
    data_dir = paths['data_dir']
    
    predictions = {}
    
    # Find all prediction files
    prediction_files = glob.glob(f"{data_dir}/predictions_*.csv")
    
    if not prediction_files:
        print(f"No prediction files found in {data_dir}")
        print("Please run model training scripts first")
        return predictions
    
    print(f"Found {len(prediction_files)} prediction files in {dataset_name}:")
    
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


def evaluate_all_models(y_true, predictions_dict, dataset_name=""):
    """
    Evaluate all models and return results
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        dataset_name (str): Name of the dataset
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for each model
    """
    results = []
    
    print(f"\n" + "="*60)
    print(f"MODEL EVALUATION RESULTS for {dataset_name}")
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
    print(f"RANKING (by R² Score) for {dataset_name}:")
    print("="*60)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i}. {row['Model']:<15} R² = {row['R2']:.4f}")
    
    return results_df


def plot_predicted_vs_actual(y_true, predictions_dict, save_path, dataset_name=""):
    """
    Create predicted vs actual scatter plots for each model
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        save_path (str): Path to save plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nCreating predicted vs actual plots for {dataset_name}...")
    
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
        title = f'{model_name} Model\nR² = {r2:.4f}'
        if dataset_name:
            title += f' ({dataset_name})'
        ax.set_title(title)
        
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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Predicted vs Actual plots saved to: {save_path}")


def plot_model_comparison(results_df, save_path, dataset_name=""):
    """
    Create comparison bar charts for model performance
    
    Args:
        results_df (pd.DataFrame): Results dataframe with metrics
        save_path (str): Path to save plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nCreating model comparison charts for {dataset_name}...")
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['MAE', 'RMSE', 'R2']
    metric_labels = ['MAE', 'RMSE', 'R²']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
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
        title = f'{label} Comparison\n{title_suffix}'
        if dataset_name:
            title += f' ({dataset_name})'
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(label)
        ax.set_xlabel('Models')
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add some padding to y-axis
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - (y_max - y_min) * 0.05, y_max + (y_max - y_min) * 0.1)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison charts saved to: {save_path}")


def plot_residuals_analysis(y_true, predictions_dict, save_path, dataset_name=""):
    """
    Create residual plots for model analysis
    
    Args:
        y_true (pd.Series): True values
        predictions_dict (dict): Dictionary of model predictions
        save_path (str): Path to save plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nCreating residual analysis plots for {dataset_name}...")

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
        title1 = f'{model_name} - Residuals vs Predicted'
        if dataset_name:
            title1 += f' ({dataset_name})'
        ax1.set_title(title1)
        ax1.grid(True, alpha=0.3)

        # Plot histogram of residuals
        ax2 = axes[1, idx]
        ax2.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='navy')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        title2 = f'{model_name} - Residuals Distribution'
        if dataset_name:
            title2 += f' ({dataset_name})'
        ax2.set_title(title2)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Residual analysis plots saved to: {save_path}")


def save_results_summary(results_df, save_path, dataset_name=""):
    """
    Save evaluation results to CSV file
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save_path (str): Path to save results
        dataset_name (str): Name of dataset
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    results_df.to_csv(save_path, index=False)
    print(f"\nEvaluation results for {dataset_name} saved to: {save_path}")


def main(dataset_name=None):
    """
    Main evaluation pipeline for individual dataset
    
    Args:
        dataset_name (str): Name of the dataset to evaluate
    """
    dataset_info = f" for {dataset_name}" if dataset_name else ""
    print("="*60)
    print(f"MODEL EVALUATION PIPELINE{dataset_info}")
    print("="*60)
    
    # Get output paths
    paths = get_output_paths(dataset_name)
    
    # Ensure chart directory exists
    os.makedirs(paths['charts_dir'], exist_ok=True)
    
    # Load true values
    y_true = load_true_values(dataset_name)
    if y_true is None:
        return
    
    # Load model predictions
    predictions_dict = load_model_predictions(dataset_name)
    if not predictions_dict:
        return
    
    # Filter out classification models from regression evaluation
    regression_predictions = {}
    for model_name, predictions in predictions_dict.items():
        if 'CLASSIFICATION' not in model_name.upper():
            regression_predictions[model_name] = predictions
    
    if not regression_predictions:
        print(f"No regression model predictions found for {dataset_name}")
        return
    
    print(f"Evaluating {len(regression_predictions)} regression models for {dataset_name}")
    
    # Evaluate all models
    results_df = evaluate_all_models(y_true, regression_predictions, dataset_name)
    
    # Create visualizations
    plot_predicted_vs_actual(y_true, regression_predictions, paths['predicted_vs_actual_path'], dataset_name)
    plot_model_comparison(results_df, paths['model_comparison_path'], dataset_name)
    plot_residuals_analysis(y_true, regression_predictions, paths['residuals_path'], dataset_name)
    
    # Save results
    save_results_summary(results_df, paths['results_path'], dataset_name)
    
    print(f"\n" + "="*60)
    print(f"EVALUATION PIPELINE COMPLETED SUCCESSFULLY for {dataset_name}!")
    print("="*60)
    print("Files created:")
    print(f"  - {paths['results_path']}")
    print(f"  - {paths['predicted_vs_actual_path']}")
    print(f"  - {paths['model_comparison_path']}")
    print(f"  - {paths['residuals_path']}")
    
    # Print final summary
    print(f"\n" + "="*60)
    print(f"FINAL SUMMARY for {dataset_name}")
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
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_name)