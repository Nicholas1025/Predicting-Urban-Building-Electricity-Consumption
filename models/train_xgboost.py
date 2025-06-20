"""
XGBoost Model Training Script for Building Energy Consumption Prediction
Trains an XGBoost regressor with hyperparameter tuning and saves results
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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


def load_preprocessed_data():
    """
    Load the preprocessed data from CSV files
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    try:
        print("Loading preprocessed data...")
        X_train = pd.read_csv("outputs/X_train.csv")
        X_test = pd.read_csv("outputs/X_test.csv")
        y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]  # First column
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]    # First column
        
        # Load feature names
        with open("outputs/feature_names.txt", "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run clean_data.py first to generate preprocessed data.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def standardize_feature_names(features):
    """Standardize feature names to avoid training/prediction mismatches"""
    # Remove special characters and standardize names
    new_columns = []
    for col in features.columns:
        new_col = col.replace('(', '').replace(')', '').replace('/', '_').replace(' ', '_')
        new_col = new_col.replace(',', '').replace('-', '_').replace('.', '_')
        new_columns.append(new_col)
    
    features.columns = new_columns
    return features

# Add this function call in load_preprocessed_data() in all individual training scripts:
# X_train = standardize_feature_names(X_train)
# X_test = standardize_feature_names(X_test)

def create_baseline_model():
    """
    Create a baseline XGBoost model with default parameters
    
    Returns:
        xgb.XGBRegressor: Baseline XGBoost model
    """
    model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    return model


def create_tuned_model():
    """
    Create an XGBoost model with manually tuned parameters
    
    Returns:
        xgb.XGBRegressor: Tuned XGBoost model
    """
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
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
        xgb.XGBRegressor: Best model from grid search
    """
    print("Performing GridSearchCV hyperparameter tuning...")
    print("This may take several minutes...")
    
    # Define parameter grid (reduced for faster execution)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    base_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0
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


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="XGBoost"):
    """
    Evaluate the model and print metrics
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features  
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
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
    
    return metrics


def plot_feature_importance(model, feature_names, save_path="outputs/charts/feature_importance_xgb.png"):
    """
    Plot and save feature importance chart
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        save_path (str): Path to save the plot
    """
    print(f"\nPlotting feature importance...")
    
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
    plt.barh(range(len(top_features)), top_features['importance'], color='lightblue', edgecolor='navy')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features - XGBoost Model')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to: {save_path}")
    
    # Print top 10 features
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.tail(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")


def save_model_and_predictions(model, y_test_pred, model_path="outputs/model_xgb.pkl", 
                             predictions_path="outputs/predictions_xgb.csv"):
    """
    Save the trained model and predictions
    
    Args:
        model: Trained model
        y_test_pred (array): Test predictions
        model_path (str): Path to save the model
        predictions_path (str): Path to save predictions
    """
    print(f"\nSaving model and predictions...")
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'predictions': y_test_pred
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to: {predictions_path}")


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


def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("XGBOOST MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    
    # Load data
    data = load_preprocessed_data()
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
    baseline_metrics = evaluate_model(baseline_model, X_train, X_test, y_train, y_test, "Baseline XGBoost")
    models_to_evaluate['Baseline'] = (baseline_model, baseline_metrics)
    
    # 2. Manually tuned model
    print("\n" + "="*50)
    print("TRAINING MANUALLY TUNED MODEL")
    print("="*50)
    tuned_model = create_tuned_model()
    tuned_model.fit(X_train, y_train)
    tuned_metrics = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, "Tuned XGBoost")
    models_to_evaluate['Tuned'] = (tuned_model, tuned_metrics)
    
    # 3. Grid search model (optional - can be commented out for faster execution)
    use_grid_search = input("\nPerform GridSearchCV? (y/n, default=n): ").lower().strip() == 'y'
    
    if use_grid_search:
        print("\n" + "="*50)
        print("TRAINING GRID SEARCH MODEL")
        print("="*50)
        grid_model = perform_grid_search(X_train, y_train)
        grid_metrics = evaluate_model(grid_model, X_train, X_test, y_train, y_test, "GridSearch XGBoost")
        models_to_evaluate['GridSearch'] = (grid_model, grid_metrics)
    
    # Select best model based on test R²
    best_model_name = max(models_to_evaluate.keys(), 
                         key=lambda k: models_to_evaluate[k][1]['test_r2'])
    best_model, best_metrics = models_to_evaluate[best_model_name]
    
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test R²: {best_metrics['test_r2']:.4f}")
    print(f"Test RMSE: {best_metrics['test_rmse']:.2f}")
    print("="*60)
    
    # Perform cross-validation on best model
    perform_cross_validation(best_model, X_train, y_train)
    
    # Plot feature importance
    plot_feature_importance(best_model, feature_names)
    
    # Save model and predictions
    save_model_and_predictions(best_model, best_metrics['test_predictions'])
    
    print(f"\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Files saved:")
    print("  - outputs/model_xgb.pkl")
    print("  - outputs/predictions_xgb.csv") 
    print("  - outputs/charts/feature_importance_xgb.png")


if __name__ == "__main__":
    main()