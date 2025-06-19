"""
Classification Models Training for Individual Dataset Analysis
Trains XGBoost, Random Forest, and SVM classifiers for energy efficiency prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        'models_dir': f"{base_dir}/models",
        'charts_dir': f"{base_dir}/charts",
        'tables_dir': f"{base_dir}/tables",
        'confusion_matrix_path': f"{base_dir}/charts/confusion_matrices_classification.png",
        'metrics_comparison_path': f"{base_dir}/charts/classification_metrics_comparison.png",
        'performance_table_path': f"{base_dir}/tables/classification_performance.csv"
    }


def load_classification_data(dataset_name=None):
    """
    Load the processed classification data for specific dataset
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: (X, y, feature_names)
    """
    paths = get_output_paths(dataset_name)
    data_dir = paths['data_dir']
    
    try:
        print(f"Loading classification data from {data_dir}...")
        
        # Try to load unified classification data first
        try:
            X = pd.read_csv(f"{data_dir}/unified_features.csv")
            y = pd.read_csv(f"{data_dir}/unified_labels.csv").iloc[:, 0]
            
            # Load feature names
            with open(f"{data_dir}/feature_names.txt", "r") as f:
                feature_names = [line.strip() for line in f.readlines()]
                
        except FileNotFoundError:
            # If unified data doesn't exist, create from training data
            print("Unified classification data not found, creating from training data...")
            X = pd.read_csv(f"{data_dir}/X_train.csv")
            X_test = pd.read_csv(f"{data_dir}/X_test.csv")
            y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]
            y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
            
            # Combine train and test data
            X = pd.concat([X, X_test], ignore_index=True)
            y_combined = pd.concat([y_train, y_test], ignore_index=True)
            
            # Create energy efficiency labels
            from preprocessing.multi_dataset_processor import create_energy_efficiency_labels
            y = create_energy_efficiency_labels(y_combined)
            
            # Save unified data for future use
            X.to_csv(f"{data_dir}/unified_features.csv", index=False)
            y.to_csv(f"{data_dir}/unified_labels.csv", index=False)
            
            feature_names = X.columns.tolist()
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Classes: {sorted(y.unique())}")
        print(f"Class distribution:")
        print(y.value_counts())
        
        return X, y, feature_names
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run data preprocessing first.")
        return None
    except Exception as e:
        print(f"Error loading classification data: {e}")
        return None


def prepare_classification_data(X, y, test_size=0.2, random_state=42):
    """
    Prepare data for classification training
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Labels
        test_size (float): Test set proportion
        random_state (int): Random state
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, label_encoder)
    """
    print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    
    # Encode string labels to numeric first
    print("Encoding string labels to numeric...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Print encoding mapping
    label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(f"Label mapping: {label_mapping}")
    
    # Split the data using encoded labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, 
        stratify=y_encoded  # Use encoded labels for stratification
    )
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder


def train_random_forest_classifier(X_train, y_train, perform_tuning=False):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("="*50)
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Tuning completed in {end_time - start_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        rf_model = grid_search.best_estimator_
    else:
        print("Training with default parameters...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return rf_model


def train_xgboost_classifier(X_train, y_train, perform_tuning=False):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels (must be numeric)
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        XGBClassifier: Trained model
    """
    print("\n" + "="*50)
    print("TRAINING XGBOOST CLASSIFIER")
    print("="*50)
    
    # Ensure labels are numeric
    if not np.issubdtype(y_train.dtype, np.integer):
        print("Warning: Converting labels to numeric for XGBoost")
        y_train = pd.Series(y_train).astype(int)
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Tuning completed in {end_time - start_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        xgb_model = grid_search.best_estimator_
    else:
        print("Training with default parameters...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return xgb_model


def train_svm_classifier(X_train, y_train, use_subset=True, subset_size=5000, perform_tuning=False):
    """
    Train SVM classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        use_subset: Whether to use subset for training (computational efficiency)
        subset_size: Size of subset if use_subset is True
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        SVC: Trained model
    """
    print("\n" + "="*50)
    print("TRAINING SVM CLASSIFIER")
    print("="*50)
    
    # Use subset for SVM if dataset is large
    if use_subset and len(X_train) > subset_size:
        print(f"Using subset of {subset_size} samples for SVM training (computational efficiency)")
        subset_indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_svm = X_train.iloc[subset_indices]
        y_train_svm = y_train[subset_indices] if hasattr(y_train, 'iloc') else y_train[subset_indices]
    else:
        X_train_svm = X_train
        y_train_svm = y_train
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'kernel': ['rbf', 'poly'],
            'C': [10, 100, 1000],
            'gamma': ['scale', 'auto']
        }
        
        svm_base = SVC(random_state=42)
        grid_search = GridSearchCV(
            svm_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train_svm, y_train_svm)
        end_time = time.time()
        
        print(f"Tuning completed in {end_time - start_time:.2f} seconds")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        svm_model = grid_search.best_estimator_
    else:
        print("Training with default parameters...")
        svm_model = SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            random_state=42
        )
        
        start_time = time.time()
        svm_model.fit(X_train_svm, y_train_svm)
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return svm_model


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder):
    """
    Evaluate classification model with all required metrics
    
    Args:
        model: Trained classifier
        X_train, X_test: Feature data
        y_train, y_test: Label data
        model_name: Name of the model
        label_encoder: Label encoder for converting back to string labels
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*50}")
    print(f"EVALUATING {model_name.upper()} CLASSIFIER")
    print(f"{'='*50}")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Weighted averages for multi-class
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Print results
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:     {test_accuracy:.4f}")
    print(f"Test Precision:    {test_precision:.4f}")
    print(f"Test Recall:       {test_recall:.4f}")
    print(f"Test F1-Score:     {test_f1:.4f}")
    
    # Detailed classification report with original labels
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test_labels, y_test_pred_labels))
    
    # Check for overfitting
    accuracy_diff = abs(train_accuracy - test_accuracy)
    print(f"\nOverfitting Check:")
    print(f"Accuracy difference (Train vs Test): {accuracy_diff:.4f}")
    
    if accuracy_diff > 0.05:
        print("⚠️  Model may be overfitting (accuracy difference > 0.05)")
    else:
        print("✅ Model shows good generalization")
    
    # Return results
    results = {
        'Model': model_name,
        'Train_Accuracy': train_accuracy,
        'Test_Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1_Score': test_f1,
        'Predictions': y_test_pred,
        'True_Labels': y_test,
        'Predictions_Text': y_test_pred_labels,
        'True_Labels_Text': y_test_labels
    }
    
    return results


def plot_confusion_matrices(all_results, label_encoder, save_path, dataset_name=""):
    """
    Plot confusion matrices for all models
    
    Args:
        all_results (list): List of evaluation results
        label_encoder: Label encoder for getting class names
        save_path (str): Path to save plot
        dataset_name (str): Name of dataset for title
    """
    print(f"\nGenerating confusion matrices for {dataset_name}...")
    
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    class_names = label_encoder.classes_
    
    for idx, results in enumerate(all_results):
        model_name = results['Model']
        y_true = results['True_Labels']
        y_pred = results['Predictions']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=axes[idx])
        
        title = f'{model_name}\nConfusion Matrix'
        if dataset_name:
            title += f' ({dataset_name})'
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
        # Rotate labels for better readability
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to: {save_path}")


def plot_classification_metrics_comparison(results_df, save_path, dataset_name=""):
    """
    Plot comparison of classification metrics across models
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save_path (str): Path to save plot
        dataset_name (str): Name of dataset for title
    """
    print(f"Creating classification metrics comparison plot for {dataset_name}...")
    
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.2
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        positions = x + i * width
        bars = ax.bar(positions, results_df[metric], width, 
                     label=label, color=colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, results_df[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    title = 'Classification Model Performance Comparison'
    if dataset_name:
        title += f' ({dataset_name})'
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classification metrics comparison saved to: {save_path}")


def save_classification_results(models, results, label_encoder, dataset_name):
    """
    Save trained classification models and predictions
    
    Args:
        models (dict): Dictionary of trained models
        results (list): List of evaluation results
        label_encoder: Label encoder
        dataset_name (str): Name of the dataset
    """
    paths = get_output_paths(dataset_name)
    
    print(f"\nSaving classification models and predictions for {dataset_name}...")
    
    # Ensure directories exist
    os.makedirs(paths['models_dir'], exist_ok=True)
    
    # Save models
    for model_name, model in models.items():
        model_filename = f"{paths['models_dir']}/model_{model_name.lower().replace(' ', '_')}_classifier.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved: {model_filename}")
    
    # Save label encoder
    joblib.dump(label_encoder, f"{paths['models_dir']}/label_encoder.pkl")
    print(f"Label encoder saved: {paths['models_dir']}/label_encoder.pkl")
    
    # Save predictions
    for result in results:
        model_name = result['Model'].lower().replace(' ', '_')
        pred_filename = f"{paths['data_dir']}/predictions_{model_name}_classification.csv"
        
        predictions_df = pd.DataFrame({
            'predictions': result['Predictions'],
            'true_labels': result['True_Labels'],
            'predictions_text': result['Predictions_Text'],
            'true_labels_text': result['True_Labels_Text']
        })
        predictions_df.to_csv(pred_filename, index=False)
        print(f"Predictions saved: {pred_filename}")


def create_professional_results_table(results_df, dataset_name):
    """
    Create professional tables for reporting
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        dataset_name (str): Name of the dataset
    """
    paths = get_output_paths(dataset_name)
    
    print(f"\nCreating professional results tables for {dataset_name}...")
    
    # Ensure directory exists
    os.makedirs(paths['tables_dir'], exist_ok=True)
    
    # Round values for better presentation
    display_df = results_df.copy()
    numeric_cols = ['Train_Accuracy', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(4)
    
    # Create clean table for reporting
    report_df = display_df[['Model', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score']].copy()
    report_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Sort by accuracy
    report_df = report_df.sort_values('Accuracy', ascending=False)
    
    # Save in CSV format
    report_df.to_csv(paths['performance_table_path'], index=False)
    
    print(f"Professional table saved to: {paths['performance_table_path']}")
    
    return report_df


def perform_cross_validation(models, X, y, cv_folds=5):
    """
    Perform cross-validation for all models
    
    Args:
        models (dict): Dictionary of trained models
        X: Features
        y: Labels
        cv_folds (int): Number of CV folds
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"\nCross-validating {model_name}...")
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
        
        cv_results[model_name] = {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Individual scores: {cv_scores}")
    
    return cv_results


def main(dataset_name=None):
    """
    Main classification training pipeline for individual dataset
    
    Args:
        dataset_name (str): Name of the dataset to process
    """
    dataset_info = f" for {dataset_name}" if dataset_name else ""
    print("="*60)
    print(f"CLASSIFICATION MODEL TRAINING PIPELINE{dataset_info}")
    print("="*60)
    
    # Get output paths
    paths = get_output_paths(dataset_name)
    
    # Load data
    data = load_classification_data(dataset_name)
    if data is None:
        print(f"❌ Failed to load classification data for {dataset_name}")
        return
    
    X, y, feature_names = data
    
    # Prepare data (includes label encoding)
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_classification_data(X, y)
    
    # Ask user about hyperparameter tuning
    perform_tuning = input("\nPerform hyperparameter tuning? (y/n, default=n): ").lower().strip() == 'y'
    
    # Train models
    models = {}
    
    # Random Forest
    models['Random Forest'] = train_random_forest_classifier(X_train, y_train, perform_tuning)
    
    # XGBoost (now with proper numeric labels)
    models['XGBoost'] = train_xgboost_classifier(X_train, y_train, perform_tuning)
    
    # SVM
    models['SVM'] = train_svm_classifier(X_train, y_train, use_subset=True, perform_tuning=perform_tuning)
    
    # Evaluate all models
    all_results = []
    for model_name, model in models.items():
        results = evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name, label_encoder)
        all_results.append(results)
    
    # Create results dataframe
    results_df = pd.DataFrame([{k: v for k, v in result.items() 
                               if k not in ['Predictions', 'True_Labels', 'Predictions_Text', 'True_Labels_Text']} 
                              for result in all_results])
    
    # Sort by test accuracy
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    
    # Generate visualizations
    plot_confusion_matrices(all_results, label_encoder, paths['confusion_matrix_path'], dataset_name)
    plot_classification_metrics_comparison(results_df, paths['metrics_comparison_path'], dataset_name)
    
    # Create professional tables
    report_df = create_professional_results_table(results_df, dataset_name)
    
    # Save models and predictions
    save_classification_results(models, all_results, label_encoder, dataset_name)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(models, X_train, y_train)
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"CLASSIFICATION TRAINING COMPLETED SUCCESSFULLY for {dataset_name}!")
    print("="*60)
    
    best_model = results_df.iloc[0]
    print(f"🏆 Best Model: {best_model['Model']}")
    print(f"   Test Accuracy: {best_model['Test_Accuracy']:.4f}")
    print(f"   Precision: {best_model['Precision']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f}")
    print(f"   F1-Score: {best_model['F1_Score']:.4f}")
    
    print(f"\n📊 Model Performance Ranking:")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"   {i}. {row['Model']:<15} (Accuracy = {row['Test_Accuracy']:.4f})")
    
    print(f"\n📁 Files Created:")
    print(f"   Models: {paths['models_dir']}/model_*_classifier.pkl")
    print(f"   Predictions: {paths['data_dir']}/predictions_*_classification.csv")
    print(f"   Charts: {paths['confusion_matrix_path']}")
    print(f"   Charts: {paths['metrics_comparison_path']}")
    print(f"   Tables: {paths['performance_table_path']}")
    
    return models, results_df, cv_results


if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_name)