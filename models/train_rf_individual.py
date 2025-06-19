"""
Random Forest Model Training Script for Individual Dataset Analysis
Trains a Random Forest regressor with hyperparameter tuning for specific dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def get_output_paths(dataset_name):
    """Get output paths for specific dataset"""
    if not dataset_name:
        base_dir = "outputs"
    else:
        base_dir = f"outputs/{dataset_name}"
    
    return {
        'data_dir': base_dir,
        'model_path': f"{base_dir}/model_rf.pkl",
        'pred_path': f"{base_dir}/predictions_rf.csv",
        'chart_path': f"{base_dir}/charts/feature_importance_rf.png",
        'tree_depth_path': f"{base_dir}/charts/tree_depth_analysis_rf.png"
    }


def load_preprocessed_data(dataset_name=None):
    """
    Load the preprocessed data from CSV files for specific dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    paths = get_output_paths(dataset_name)
    data_dir = paths['data_dir']
    
    try:
        print(f"Loading preprocessed data from {data_dir}...")
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
        
        # Load feature names
        with open(f"{data_dir}/feature_names.txt", "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data preprocessing first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_baseline_model():
    """
    Create a baseline Random Forest model with default parameters
    
    Returns:
        RandomForestRegressor: Baseline Random Forest model
    """
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )
    return model


def create_tuned_model():
    """
    Create a Random Forest model with manually tuned parameters
    
    Returns:
        RandomForestRegressor: Tuned Random Forest model
    """
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    return model


def perform_grid_search(X_train, y_train, cv_folds=3):
    """
    Perform GridSearchCV to find optimal hyperparameters
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        RandomForestRegressor: Best model from grid search
    """
    print("Performing GridSearchCV hyperparameter tuning...")
    print("This may take several minutes...")
    
    # Define parameter grid (reduced for faster execution)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Grid search completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (RMSE): {-grid_search.best_score_:.2f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Random Forest"):
    """
    Evaluate the model and print metrics
    
    Args:
        model: Trained model
        X_train, X_test: Feature data
        y_train, y_test: Target data
        model_name (str): Name of the model for printing
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"EVALUATING {model_name.upper()} MODEL")
    print(f"{'='*50}")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"Training Set Metrics:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # Print OOB score if available
    if hasattr(model, 'oob_score_') and model.oob_score_:
        print(f"\nOut-of-Bag Score: {model.oob_score_:.4f}")
    
    # Check for overfitting
    print(f"\nOverfitting Check:")
    print(f"  RMSE difference (Train vs Test): {abs(train_rmse - test_rmse):.2f}")
    print(f"  R² difference (Train vs Test): {abs(train_r2 - test_r2):.4f}")
    
    if train_r2 - test_r2 > 0.1:
        print("  ⚠️  Model may be overfitting (R² difference > 0.1)")
    else:
        print("  ✅ Model shows good generalization")
    
    # Return metrics dictionary
    metrics = {
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_r2': train_r2,
        'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2,
        'test_predictions': y_test_pred
    }
    
    if hasattr(model, 'oob_score_') and model.oob_score_:
        metrics['oob_score'] = model.oob_score_
    
    return metrics


def plot_feature_importance(model, feature_names, save_path, dataset_name=""):
    """
    Plot and save feature importance chart
    
    Args:
        model: Trained Random Forest model
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nPlotting feature importance for {dataset_name}...")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Plot top 20 features
    top_features = importance_df.tail(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='lightgreen', edgecolor='darkgreen')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    title = f'Top 20 Most Important Features - Random Forest Model'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {save_path}")
    
    # Print top 10 features
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.tail(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")


def plot_tree_depth_analysis(model, save_path, dataset_name=""):
    """
    Plot tree depth analysis for Random Forest
    
    Args:
        model: Trained Random Forest model
        save_path (str): Path to save the plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nAnalyzing tree depths for {dataset_name}...")
    
    # Get depths of all trees
    tree_depths = [tree.tree_.max_depth for tree in model.estimators_]
    
    plt.figure(figsize=(10, 6))
    plt.hist(tree_depths, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Tree Depth')
    plt.ylabel('Frequency')
    title = f'Distribution of Tree Depths in Random Forest'
    if dataset_name:
        title += f' ({dataset_name})'
    plt.title(title)
    plt.axvline(np.mean(tree_depths), color='red', linestyle='--', 
                label=f'Mean Depth: {np.mean(tree_depths):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Tree depth analysis plot saved to: {save_path}")
    print(f"Average tree depth: {np.mean(tree_depths):.1f}")
    print(f"Max tree depth: {np.max(tree_depths)}")
    print(f"Min tree depth: {np.min(tree_depths)}")


def save_model_and_predictions(model, y_test_pred, dataset_name):
    """
    Save the trained model and predictions
    
    Args:
        model: Trained model
        y_test_pred (array): Test predictions
        dataset_name (str): Name of the dataset
    """
    paths = get_output_paths(dataset_name)
    
    print(f"\nSaving model and predictions for {dataset_name}...")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
    
    # Save model
    joblib.dump(model, paths['model_path'])
    print(f"Model saved to: {paths['model_path']}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'predictions': y_test_pred
    })
    predictions_df.to_csv(paths['pred_path'], index=False)
    print(f"Predictions saved to: {paths['pred_path']}")


def perform_cross_validation(model, X_train, y_train, cv_folds=5):
    """
    Perform cross-validation to assess model stability
    
    Args:
        model: Model to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=-1)
    cv_rmse_scores = -cv_scores
    
    print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
    print(f"Mean CV RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")


def main(dataset_name=None):
    """
    Main training pipeline for individual dataset
    
    Args:
        dataset_name (str): Name of the dataset to process
    """
    dataset_info = f" for {dataset_name}" if dataset_name else ""
    print("="*60)
    print(f"RANDOM FOREST MODEL TRAINING PIPELINE{dataset_info}")
    print("="*60)
    
    # Get output paths
    paths = get_output_paths(dataset_name)
    
    # Load data
    data = load_preprocessed_data(dataset_name)
    if data is None:
        return
    
    X_train, X_test, y_train, y_test, feature_names = data
    
    # Train different model variants
    models_to_evaluate = {}
    
    # 1. Baseline model
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)
    baseline_model = create_baseline_model()
    baseline_model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_train, X_test, y_train, y_test, "Baseline Random Forest")
    models_to_evaluate['Baseline'] = (baseline_model, baseline_metrics)
    
    # 2. Manually tuned model
    print("\n" + "="*50)
    print("TRAINING MANUALLY TUNED MODEL")
    print("="*50)
    tuned_model = create_tuned_model()
    tuned_model.fit(X_train, y_train)
    tuned_metrics = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, "Tuned Random Forest")
    models_to_evaluate['Tuned'] = (tuned_model, tuned_metrics)
    
    # 3. Grid search model (optional)
    use_grid_search = input("\nPerform GridSearchCV? (y/n, default=n): ").lower().strip() == 'y'
    
    if use_grid_search:
        print("\n" + "="*50)
        print("TRAINING GRID SEARCH MODEL")
        print("="*50)
        grid_model = perform_grid_search(X_train, y_train)
        grid_metrics = evaluate_model(grid_model, X_train, X_test, y_train, y_test, "GridSearch Random Forest")
        models_to_evaluate['GridSearch'] = (grid_model, grid_metrics)
    
    # Select best model based on test R²
    best_model_name = max(models_to_evaluate.keys(), 
                         key=lambda k: models_to_evaluate[k][1]['test_r2'])
    best_model, best_metrics = models_to_evaluate[best_model_name]
    
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {best_metrics['test_r2']:.4f}")
    print(f"Test RMSE: {best_metrics['test_rmse']:.2f}")
    if 'oob_score' in best_metrics:
        print(f"OOB Score: {best_metrics['oob_score']:.4f}")
    print("="*60)
    
    # Perform cross-validation on best model
    perform_cross_validation(best_model, X_train, y_train)
    
    # Plot feature importance
    plot_feature_importance(best_model, feature_names, paths['chart_path'], dataset_name)
    
    # Plot tree depth analysis
    plot_tree_depth_analysis(best_model, paths['tree_depth_path'], dataset_name)
    
    # Save model and predictions
    save_model_and_predictions(best_model, best_metrics['test_predictions'], dataset_name)
    
    print(f"\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print(f"  - {paths['model_path']}")
    print(f"  - {paths['pred_path']}")
    print(f"  - {paths['chart_path']}")
    print(f"  - {paths['tree_depth_path']}")


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_name)