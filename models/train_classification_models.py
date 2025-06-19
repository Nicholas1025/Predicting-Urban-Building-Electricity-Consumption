"""
Classification Models Training for Building Energy Efficiency Prediction
Trains XGBoost, Random Forest, and SVM classifiers to predict energy efficiency labels
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


def load_classification_data():
    """
    Load the processed classification data
    
    Returns:
        tuple: (X, y, feature_names)
    """
    try:
        print("Loading classification data...")
        X = pd.read_csv("outputs/unified_features.csv")
        y = pd.read_csv("outputs/unified_labels.csv").iloc[:, 0]
        
        # Load feature names
        with open("outputs/feature_names.txt", "r") as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Classes: {sorted(y.unique())}")
        print(f"Class distribution:")
        print(y.value_counts())
        
        return X, y, feature_names
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run multi_dataset_processor.py first to generate processed data.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
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
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=y  # Ensure balanced splits
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
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


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
        y_train: Training labels
        perform_tuning: Whether to perform hyperparameter tuning
        
    Returns:
        XGBClassifier: Trained model
    """
    print("\n" + "="*50)
    print("TRAINING XGBOOST CLASSIFIER")
    print("="*50)
    
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
        y_train_svm = y_train.iloc[subset_indices]
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


def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate classification model with all required metrics
    
    Args:
        model: Trained classifier
        X_train, X_test: Feature data
        y_train, y_test: Label data
        model_name: Name of the model
        
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
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
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
        'True_Labels': y_test
    }
    
    return results


def plot_confusion_matrices(all_results, save_dir="outputs/charts"):
    """
    Plot confusion matrices for all models
    
    Args:
        all_results (list): List of evaluation results
        save_dir (str): Directory to save plots
    """
    print("\nGenerating confusion matrices...")
    
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, results in enumerate(all_results):
        model_name = results['Model']
        y_true = results['True_Labels']
        y_pred = results['Predictions']
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sorted(y_true.unique()),
                   yticklabels=sorted(y_true.unique()),
                   ax=axes[idx])
        
        axes[idx].set_title(f'{model_name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        
        # Rotate labels for better readability
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrices_classification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to: {save_dir}/confusion_matrices_classification.png")


def plot_classification_metrics_comparison(results_df, save_dir="outputs/charts"):
    """
    Plot comparison of classification metrics across models
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save_dir (str): Directory to save plots
    """
    print("Creating classification metrics comparison plot...")
    
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
    ax.set_title('Classification Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/classification_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Classification metrics comparison saved to: {save_dir}/classification_metrics_comparison.png")


def save_classification_models_and_predictions(models, results, save_dir="outputs"):
    """
    Save trained classification models and predictions
    
    Args:
        models (dict): Dictionary of trained models
        results (list): List of evaluation results
        save_dir (str): Directory to save files
    """
    print(f"\nSaving classification models and predictions...")
    
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    
    # Save models
    for model_name, model in models.items():
        model_filename = f"{save_dir}/models/model_{model_name.lower().replace(' ', '_')}_classifier.pkl"
        joblib.dump(model, model_filename)
        print(f"Model saved: {model_filename}")
    
    # Save predictions
    for result in results:
        model_name = result['Model'].lower().replace(' ', '_')
        pred_filename = f"{save_dir}/predictions_{model_name}_classification.csv"
        
        predictions_df = pd.DataFrame({
            'predictions': result['Predictions'],
            'true_labels': result['True_Labels']
        })
        predictions_df.to_csv(pred_filename, index=False)
        print(f"Predictions saved: {pred_filename}")


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


def create_professional_results_table(results_df, save_dir="outputs/tables"):
    """
    Create professional tables for reporting
    
    Args:
        results_df (pd.DataFrame): Results dataframe
        save_dir (str): Directory to save tables
    """
    print("\nCreating professional results tables...")
    
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # Save in multiple formats
    # CSV format
    report_df.to_csv(f"{save_dir}/classification_performance.csv", index=False)
    
    # Markdown format
    with open(f"{save_dir}/classification_performance_markdown.txt", "w") as f:
        f.write("# Classification Model Performance Comparison\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for _, row in report_df.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} |\n")
    
    # LaTeX format
    with open(f"{save_dir}/classification_performance_latex.txt", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Classification Model Performance Comparison}\n")
        f.write("\\begin{tabular}{|l|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Model & Accuracy & Precision & Recall & F1-Score \\\\\n")
        f.write("\\hline\n")
        for _, row in report_df.iterrows():
            f.write(f"{row['Model']} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1-Score']:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Formatted text for easy copying
    with open(f"{save_dir}/classification_performance_formatted.txt", "w") as f:
        f.write("CLASSIFICATION MODEL PERFORMANCE COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        
        col_widths = [15, 10, 11, 8, 10]
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Header
        header_line = ""
        for header, width in zip(headers, col_widths):
            header_line += f"{header:<{width}}"
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        # Data rows
        for _, row in report_df.iterrows():
            data_line = ""
            data_line += f"{row['Model']:<{col_widths[0]}}"
            data_line += f"{row['Accuracy']:<{col_widths[1]}.4f}"
            data_line += f"{row['Precision']:<{col_widths[2]}.4f}"
            data_line += f"{row['Recall']:<{col_widths[3]}.4f}"
            data_line += f"{row['F1-Score']:<{col_widths[4]}.4f}"
            f.write(data_line + "\n")
    
    print("Professional tables saved:")
    print(f"  - CSV: {save_dir}/classification_performance.csv")
    print(f"  - Markdown: {save_dir}/classification_performance_markdown.txt")
    print(f"  - LaTeX: {save_dir}/classification_performance_latex.txt")
    print(f"  - Formatted: {save_dir}/classification_performance_formatted.txt")
    
    return report_df


def main():
    """
    Main classification training pipeline
    """
    print("="*60)
    print("CLASSIFICATION MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    
    # Load data
    data = load_classification_data()
    if data is None:
        print("❌ Failed to load data. Please run multi_dataset_processor.py first.")
        return
    
    X, y, feature_names = data
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_classification_data(X, y)
    
    # Ask user about hyperparameter tuning
    perform_tuning = input("\nPerform hyperparameter tuning? (y/n, default=n): ").lower().strip() == 'y'
    
    # Train models
    models = {}
    
    # Random Forest
    models['Random Forest'] = train_random_forest_classifier(X_train, y_train, perform_tuning)
    
    # XGBoost
    models['XGBoost'] = train_xgboost_classifier(X_train, y_train, perform_tuning)
    
    # SVM
    models['SVM'] = train_svm_classifier(X_train, y_train, use_subset=True, perform_tuning=perform_tuning)
    
    # Evaluate all models
    all_results = []
    for model_name, model in models.items():
        results = evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name)
        all_results.append(results)
    
    # Create results dataframe
    results_df = pd.DataFrame([{k: v for k, v in result.items() if k not in ['Predictions', 'True_Labels']} 
                              for result in all_results])
    
    # Sort by test accuracy
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    
    # Generate visualizations
    plot_confusion_matrices(all_results)
    plot_classification_metrics_comparison(results_df)
    
    # Create professional tables
    report_df = create_professional_results_table(results_df)
    
    # Save models and predictions
    save_classification_models_and_predictions(models, all_results)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(models, X_train, y_train)
    
    # Final summary
    print(f"\n" + "="*60)
    print("CLASSIFICATION TRAINING COMPLETED SUCCESSFULLY!")
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
    print(f"   Models: outputs/models/model_*_classifier.pkl")
    print(f"   Predictions: outputs/predictions_*_classification.csv")
    print(f"   Charts: outputs/charts/confusion_matrices_classification.png")
    print(f"   Charts: outputs/charts/classification_metrics_comparison.png")
    print(f"   Tables: outputs/tables/classification_performance.*")
    
    return models, results_df, cv_results


if __name__ == "__main__":
    main()