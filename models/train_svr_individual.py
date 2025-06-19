"""
SVR Model Training Script for Individual Dataset Analysis
Trains a Support Vector Regression model with hyperparameter tuning for specific dataset
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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
        'model_path': f"{base_dir}/model_svr.pkl",
        'pred_path': f"{base_dir}/predictions_svr.csv",
        'scaler_path': f"{base_dir}/scaler_svr.pkl",
        'support_vectors_path': f"{base_dir}/charts/support_vectors_analysis_svr.png",
        'kernel_path': f"{base_dir}/charts/kernel_analysis_svr.png"
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


def prepare_data_for_svr(X_train, X_test, y_train, y_test, use_subset=True, subset_size=5000):
    """
    Prepare data for SVR training with optional subset sampling and scaling
    
    Args:
        X_train, X_test, y_train, y_test: Original training and test data
        use_subset (bool): Whether to use a subset for training
        subset_size (int): Size of training subset if use_subset is True
        
    Returns:
        tuple: Prepared data and scaler
    """
    print(f"Preparing data for SVR training...")
    
    # Use subset for training if dataset is large (SVR doesn't scale well)
    if use_subset and len(X_train) > subset_size:
        print(f"Using subset of {subset_size} samples for SVR training (computational efficiency)")
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_svr = X_train.iloc[subset_indices].copy()
        y_train_svr = y_train.iloc[subset_indices].copy()
    else:
        X_train_svr = X_train.copy()
        y_train_svr = y_train.copy()
    
    # SVR benefits from feature scaling - apply additional scaling if needed
    scaler = None
    try:
        # Try to load existing scaler
        scaler = joblib.load(f"{get_output_paths(None)['data_dir']}/scaler.pkl")
        print("Using existing scaler from preprocessing")
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_svr), 
                                     columns=X_train_svr.columns, 
                                     index=X_train_svr.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, 
                                    index=X_test.index)
    except FileNotFoundError:
        # Create new scaler if not found
        print("Creating new scaler for SVR")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_svr), 
                                     columns=X_train_svr.columns, 
                                     index=X_train_svr.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                    columns=X_test.columns, 
                                    index=X_test.index)
    
    print(f"Final training data shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train_svr, y_test, scaler


def create_baseline_model():
    """
    Create a baseline SVR model with default parameters
    
    Returns:
        SVR: Baseline SVR model
    """
    model = SVR(
        kernel='rbf',
        cache_size=1000  # Increase cache for better performance
    )
    return model


def create_tuned_model():
    """
    Create an SVR model with manually tuned parameters
    
    Returns:
        SVR: Tuned SVR model
    """
    model = SVR(
        kernel='rbf',
        C=100,
        gamma='scale',
        epsilon=0.1,
        cache_size=1000
    )
    return model


def perform_grid_search(X_train, y_train, cv_folds=3):
    """
    Perform GridSearchCV to find optimal hyperparameters
    Note: This can be very time-consuming for SVR
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
        
    Returns:
        SVR: Best model from grid search
    """
    print("Performing GridSearchCV hyperparameter tuning...")
    print("This may take a very long time for SVR...")
    
    # Define parameter grid (reduced for faster execution)
    param_grid = {
        'kernel': ['rbf', 'poly'],
        'C': [10, 100, 1000],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 1.0]
    }
    
    base_model = SVR(cache_size=1000)
    
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


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="SVR"):
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
    
    # SVR-specific information
    print(f"\nSVR Model Information:")
    print(f"  Kernel: {model.kernel}")
    print(f"  Number of Support Vectors: {len(model.support_vectors_)}")
    print(f"  Support Vector Ratio: {len(model.support_vectors_) / len(X_train):.2%}")
    
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
        'test_predictions': y_test_pred,
        'n_support_vectors': len(model.support_vectors_),
        'support_vector_ratio': len(model.support_vectors_) / len(X_train)
    }
    
    return metrics


def plot_support_vectors_analysis(model, X_train, save_path, dataset_name=""):
    """
    Plot support vectors analysis
    
    Args:
        model: Trained SVR model
        X_train (pd.DataFrame): Training features
        save_path (str): Path to save the plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nAnalyzing support vectors for {dataset_name}...")
    
    n_features = X_train.shape[1]
    n_support_vectors = len(model.support_vectors_)
    support_ratio = n_support_vectors / len(X_train)
    
    # Create support vectors analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Support vector statistics
    categories = ['Training Samples', 'Support Vectors']
    values = [len(X_train), n_support_vectors]
    colors = ['lightcoral', 'lightblue']
    
    ax1.bar(categories, values, color=colors, edgecolor='darkblue')
    ax1.set_ylabel('Count')
    title1 = 'Training Samples vs Support Vectors'
    if dataset_name:
        title1 += f' ({dataset_name})'
    ax1.set_title(title1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add percentage annotation
    ax1.text(1, n_support_vectors + len(X_train)*0.05, 
             f'{support_ratio:.1%}', ha='center', fontweight='bold')
    
    # Plot 2: Support vector distribution (first 2 features for visualization)
    if n_features >= 2:
        support_indices = model.support_
        
        ax2.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], 
                   c='lightgray', alpha=0.6, s=20, label='Training Points')
        ax2.scatter(X_train.iloc[support_indices, 0], X_train.iloc[support_indices, 1], 
                   c='red', alpha=0.8, s=50, label='Support Vectors', edgecolors='black')
        ax2.set_xlabel(f'Feature: {X_train.columns[0]}')
        ax2.set_ylabel(f'Feature: {X_train.columns[1]}')
        title2 = 'Support Vectors Distribution (First 2 Features)'
        if dataset_name:
            title2 += f' ({dataset_name})'
        ax2.set_title(title2)
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Not enough features\nfor 2D visualization', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Support Vectors Distribution')
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Support vectors analysis plot saved to: {save_path}")
    print(f"Number of support vectors: {n_support_vectors}")
    print(f"Support vector ratio: {support_ratio:.2%}")


def plot_kernel_analysis(model, save_path, dataset_name=""):
    """
    Plot kernel analysis information
    
    Args:
        model: Trained SVR model
        save_path (str): Path to save the plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nAnalyzing kernel parameters for {dataset_name}...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create a text-based visualization of model parameters
    params_text = f"""
SVR Model Parameters{f' ({dataset_name})' if dataset_name else ''}:

Kernel: {model.kernel}
C (Regularization): {model.C}
Gamma: {model.gamma}
Epsilon: {model.epsilon}

Model Statistics:
Support Vectors: {len(model.support_vectors_)}
Dual Coefficients Shape: {model.dual_coef_.shape}
Intercept: {model.intercept_[0]:.4f}
"""
    
    ax.text(0.1, 0.5, params_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    title = 'SVR Model Parameters and Statistics'
    if dataset_name:
        title += f' ({dataset_name})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Kernel analysis plot saved to: {save_path}")


def save_model_and_predictions(model, y_test_pred, scaler, dataset_name):
    """
    Save the trained model, predictions, and scaler
    
    Args:
        model: Trained model
        y_test_pred (array): Test predictions
        scaler: Scaler used for preprocessing
        dataset_name (str): Name of the dataset
    """
    paths = get_output_paths(dataset_name)
    
    print(f"\nSaving model and predictions for {dataset_name}...")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
    
    # Save model
    joblib.dump(model, paths['model_path'])
    print(f"Model saved to: {paths['model_path']}")
    
    # Save scaler if different from original
    if scaler is not None:
        joblib.dump(scaler, paths['scaler_path'])
        print(f"SVR scaler saved to: {paths['scaler_path']}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'predictions': y_test_pred
    })
    predictions_df.to_csv(paths['pred_path'], index=False)
    print(f"Predictions saved to: {paths['pred_path']}")


def perform_cross_validation(model, X_train, y_train, cv_folds=3):
    """
    Perform cross-validation to assess model stability
    Note: Using fewer folds for SVR due to computational cost
    
    Args:
        model: Model to evaluate
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        cv_folds (int): Number of cross-validation folds
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    print("This may take some time for SVR...")
    
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
    print(f"SVR MODEL TRAINING PIPELINE{dataset_info}")
    print("="*60)
    
    # Get output paths
    paths = get_output_paths(dataset_name)
    
    # Load data
    data = load_preprocessed_data(dataset_name)
    if data is None:
        return
    
    X_train, X_test, y_train, y_test, feature_names = data
    
    # Prepare data for SVR (with scaling and optional subset)
    use_subset = True
    subset_size = 5000
    
    if len(X_train) > subset_size:
        use_subset_response = input(f"\nDataset has {len(X_train)} samples. Use subset of {subset_size} for SVR training? (y/n, default=y): ").lower().strip()
        if use_subset_response == 'n':
            use_subset = False
    
    X_train_svr, X_test_svr, y_train_svr, y_test, scaler = prepare_data_for_svr(
        X_train, X_test, y_train, y_test, use_subset, subset_size)
    
    # Train different model variants
    models_to_evaluate = {}
    
    # 1. Baseline model
    print("\n" + "="*50)
    print("TRAINING BASELINE MODEL")
    print("="*50)
    baseline_model = create_baseline_model()
    start_time = time.time()
    baseline_model.fit(X_train_svr, y_train_svr)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    baseline_metrics = evaluate_model(baseline_model, X_train_svr, X_test_svr, y_train_svr, y_test, "Baseline SVR")
    models_to_evaluate['Baseline'] = (baseline_model, baseline_metrics)
    
    # 2. Manually tuned model
    print("\n" + "="*50)
    print("TRAINING MANUALLY TUNED MODEL")
    print("="*50)
    tuned_model = create_tuned_model()
    start_time = time.time()
    tuned_model.fit(X_train_svr, y_train_svr)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    tuned_metrics = evaluate_model(tuned_model, X_train_svr, X_test_svr, y_train_svr, y_test, "Tuned SVR")
    models_to_evaluate['Tuned'] = (tuned_model, tuned_metrics)
    
    # 3. Grid search model (optional - can be very slow for SVR)
    use_grid_search = input("\nPerform GridSearchCV? (WARNING: Very slow for SVR) (y/n, default=n): ").lower().strip() == 'y'
    
    if use_grid_search:
        print("\n" + "="*50)
        print("TRAINING GRID SEARCH MODEL")
        print("="*50)
        grid_model = perform_grid_search(X_train_svr, y_train_svr)
        grid_metrics = evaluate_model(grid_model, X_train_svr, X_test_svr, y_train_svr, y_test, "GridSearch SVR")
        models_to_evaluate['GridSearch'] = (grid_model, grid_metrics)
    
    # Select best model based on test R²
    best_model_name = max(models_to_evaluate.keys(), 
                         key=lambda k: models_to_evaluate[k][1]['test_r2'])
    best_model, best_metrics = models_to_evaluate[best_model_name]
    
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {best_metrics['test_r2']:.4f}")
    print(f"Test RMSE: {best_metrics['test_rmse']:.2f}")
    print(f"Support Vectors: {best_metrics['n_support_vectors']}")
    print(f"Support Vector Ratio: {best_metrics['support_vector_ratio']:.2%}")
    print("="*60)
    
    # Perform cross-validation on best model (with fewer folds for SVR)
    perform_cross_validation(best_model, X_train_svr, y_train_svr, cv_folds=3)
    
    # Plot support vectors analysis
    plot_support_vectors_analysis(best_model, X_train_svr, paths['support_vectors_path'], dataset_name)
    
    # Plot kernel analysis
    plot_kernel_analysis(best_model, paths['kernel_path'], dataset_name)
    
    # Save model and predictions
    save_model_and_predictions(best_model, best_metrics['test_predictions'], scaler, dataset_name)
    
    print(f"\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print(f"  - {paths['model_path']}")
    print(f"  - {paths['pred_path']}")
    print(f"  - {paths['scaler_path']}")
    print(f"  - {paths['support_vectors_path']}")
    print(f"  - {paths['kernel_path']}")


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_name)