"""
Cross-Year Analysis for Building Energy Consumption Models
Tests unified models on individual year datasets to analyze temporal performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_individual_dataset(file_path, year):
    """
    Load and preprocess individual year dataset
    
    Args:
        file_path (str): Path to dataset
        year (str): Year identifier
        
    Returns:
        tuple: (features, target, labels, info)
    """
    print(f"Loading {year} dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Raw shape: {df.shape}")
        
        # Apply same preprocessing as unified dataset
        from preprocessing.multi_dataset_processor import (
            standardize_column_names, create_unified_target_variable, 
            create_energy_efficiency_labels
        )
        
        # Standardize columns
        df_std = standardize_column_names(df)
        
        try:
            df_with_target, target_col = create_unified_target_variable(df_std)

            if target_col not in df_with_target.columns:
                raise ValueError(f"Target column {target_col} not created successfully")
                

            if isinstance(df_with_target[target_col], pd.DataFrame):
                print(f"Warning: Target variable is DataFrame, taking first column")
                df_with_target[target_col] = df_with_target[target_col].iloc[:, 0]
            
        except Exception as e:
            print(f"Error in target variable creation: {e}")

            numerical_cols = df_std.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                target_col = numerical_cols[0]
                df_with_target = df_std.copy()
                df_with_target['EnergyConsumption'] = df_with_target[target_col]
                target_col = 'EnergyConsumption'
                print(f"Fallback: Using {numerical_cols[0]} as target variable")
            else:
                print(f"No suitable target variable found in {year} dataset")
                return None, None, None, None
        
        # Remove invalid rows
        valid_mask = ~(df_with_target[target_col].isnull() | (df_with_target[target_col] <= 0))
        df_clean = df_with_target[valid_mask].copy()
        
        if len(df_clean) == 0:
            print(f"No valid data remaining after cleaning for {year}")
            return None, None, None, None
        
        # Create classification labels
        labels = create_energy_efficiency_labels(df_clean[target_col])
        
        # Separate features and target
        target = df_clean[target_col].copy()
        features = df_clean.drop(columns=[target_col]).copy()
        
        # Drop irrelevant columns (same as unified preprocessing)
        columns_to_drop = [
            'OSEBuildingID', 'BuildingName', 'Address', 'City', 'State',
            'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 
            'TaxParcelIdentificationNumber', 'ComplianceStatus', 
            'Comments', 'DefaultData', 'ListOfAllPropertyUseTypes',
            'EnergyConsumption'   
        ]
        
        existing_cols_to_drop = [col for col in columns_to_drop if col in features.columns]
        if existing_cols_to_drop:
            features = features.drop(columns=existing_cols_to_drop)
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if features[col].nunique() > 50:
                features = features.drop(columns=[col])
            else:
                le = LabelEncoder()
                features[col] = features[col].fillna('Unknown')
                try:
                    features[col] = le.fit_transform(features[col].astype(str))
                except Exception as e:
                    print(f"Warning: Could not encode {col}, dropping it: {e}")
                    features = features.drop(columns=[col])
        
        # Handle missing values
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            features[numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].median())
        
        # Remove constant columns
        constant_cols = [col for col in features.columns if features[col].nunique() <= 1]
        if constant_cols:
            features = features.drop(columns=constant_cols)
        
        print(f"  Processed shape: {features.shape}")
        print(f"  Target samples: {len(target)}")
        print(f"  Valid samples: {len(labels)}")
        
        info = {
            'year': year,
            'samples': len(features),
            'features': len(features.columns),
            'target_range': (target.min(), target.max()),
            'class_distribution': labels.value_counts().to_dict()
        }
        
        return features, target, labels, info
        
    except Exception as e:
        print(f"Error processing {year} dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def align_features_with_unified(features, unified_feature_names):
    """
    Align individual dataset features with unified model features
    
    Args:
        features (pd.DataFrame): Individual dataset features
        unified_feature_names (list): Feature names from unified model
        
    Returns:
        pd.DataFrame: Aligned features
    """
    # Find common features
    common_features = [col for col in unified_feature_names if col in features.columns]
    missing_features = [col for col in unified_feature_names if col not in features.columns]
    
    if missing_features:
        print(f"  Missing features: {len(missing_features)} (will be filled with zeros)")
        # Add missing features with zero values
        for feature in missing_features:
            features[feature] = 0
    
    # Select and reorder features to match unified model
    aligned_features = features[unified_feature_names].copy()
    
    print(f"  Features aligned: {aligned_features.shape}")
    
    return aligned_features


def load_trained_models():
    """
    Load trained regression and classification models
    
    Returns:
        tuple: (regression_models, classification_models)
    """
    print("Loading trained models...")
    
    regression_models = {}
    classification_models = {}
    
    model_files = {
        'XGBoost': 'outputs/model_xgb.pkl',
        'Random Forest': 'outputs/model_rf.pkl',
        'SVR': 'outputs/model_svr.pkl'
    }
    
    classification_files = {
        'XGBoost': 'outputs/models/model_xgboost_classifier.pkl',
        'Random Forest': 'outputs/models/model_random_forest_classifier.pkl',
        'SVM': 'outputs/models/model_svm_classifier.pkl'
    }
    
    # Load regression models
    for model_name, file_path in model_files.items():
        try:
            model = joblib.load(file_path)
            regression_models[model_name] = model
            print(f"  ✅ Loaded {model_name} regression model")
        except FileNotFoundError:
            print(f"  ❌ {model_name} regression model not found: {file_path}")
    
    # Load classification models
    for model_name, file_path in classification_files.items():
        try:
            model = joblib.load(file_path)
            classification_models[model_name] = model
            print(f"  ✅ Loaded {model_name} classification model")
        except FileNotFoundError:
            print(f"  ❌ {model_name} classification model not found: {file_path}")
    
    return regression_models, classification_models


def test_regression_models_on_year(models, features, target, year, scaler=None):
    """
    Test regression models on specific year data
    
    Args:
        models (dict): Trained regression models
        features (pd.DataFrame): Features for the year
        target (pd.Series): Target values for the year
        year (str): Year identifier
        scaler: Fitted scaler (if available)
        
    Returns:
        list: List of results dictionaries
    """
    print(f"\nTesting regression models on {year} data...")
    
    results = []
    
    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = pd.DataFrame(
            scaler.transform(features), 
            columns=features.columns, 
            index=features.index
        )
    else:
        features_scaled = features
    
    for model_name, model in models.items():
        try:
            # Make predictions
            predictions = model.predict(features_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(target, predictions))
            mae = mean_absolute_error(target, predictions)
            r2 = r2_score(target, predictions)
            
            result = {
                'Model': model_name,
                'Year': year,
                'RMSE': rmse,
                'MAE': mae,
                'R2_Score': r2,
                'Samples': len(target),
                'Task': 'Regression'
            }
            
            results.append(result)
            
            print(f"  {model_name}: R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
    
    return results


def test_classification_models_on_year(models, features, labels, year, scaler=None):
    """
    Test classification models on specific year data
    
    Args:
        models (dict): Trained classification models
        features (pd.DataFrame): Features for the year
        labels (pd.Series): Labels for the year
        year (str): Year identifier
        scaler: Fitted scaler (if available)
        
    Returns:
        list: List of results dictionaries
    """
    print(f"\nTesting classification models on {year} data...")
    
    results = []
    
    # Scale features if scaler is available
    if scaler is not None:
        features_scaled = pd.DataFrame(
            scaler.transform(features), 
            columns=features.columns, 
            index=features.index
        )
    else:
        features_scaled = features
    
    for model_name, model in models.items():
        try:
            # Make predictions
            predictions = model.predict(features_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            
            result = {
                'Model': model_name,
                'Year': year,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Samples': len(labels),
                'Task': 'Classification'
            }
            
            results.append(result)
            
            print(f"  {model_name}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")
    
    return results


def create_cross_year_performance_plots(regression_results, classification_results, save_dir="outputs/charts"):
    """
    Create cross-year performance visualization plots
    
    Args:
        regression_results (list): Regression results
        classification_results (list): Classification results
        save_dir (str): Directory to save plots
    """
    print("\nCreating cross-year performance plots...")
    
    # Convert to DataFrames
    reg_df = pd.DataFrame(regression_results)
    class_df = pd.DataFrame(classification_results)
    
    if len(reg_df) == 0 and len(class_df) == 0:
        print("No results to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Regression R² across years
    if len(reg_df) > 0:
        reg_pivot = reg_df.pivot(index='Year', columns='Model', values='R2_Score')
        reg_pivot.plot(kind='bar', ax=axes[0,0], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0,0].set_title('Regression R² Score Across Years')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].legend(title='Models')
        axes[0,0].tick_params(axis='x', rotation=0)
        axes[0,0].grid(axis='y', alpha=0.3)
    
    # Regression RMSE across years
    if len(reg_df) > 0:
        rmse_pivot = reg_df.pivot(index='Year', columns='Model', values='RMSE')
        rmse_pivot.plot(kind='bar', ax=axes[0,1], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0,1].set_title('Regression RMSE Across Years')
        axes[0,1].set_ylabel('RMSE')
        axes[0,1].legend(title='Models')
        axes[0,1].tick_params(axis='x', rotation=0)
        axes[0,1].grid(axis='y', alpha=0.3)
    
    # Classification Accuracy across years
    if len(class_df) > 0:
        acc_pivot = class_df.pivot(index='Year', columns='Model', values='Accuracy')
        acc_pivot.plot(kind='bar', ax=axes[1,0], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[1,0].set_title('Classification Accuracy Across Years')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend(title='Models')
        axes[1,0].tick_params(axis='x', rotation=0)
        axes[1,0].grid(axis='y', alpha=0.3)
    
    # Classification F1-Score across years
    if len(class_df) > 0:
        f1_pivot = class_df.pivot(index='Year', columns='Model', values='F1_Score')
        f1_pivot.plot(kind='bar', ax=axes[1,1], color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[1,1].set_title('Classification F1-Score Across Years')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].legend(title='Models')
        axes[1,1].tick_params(axis='x', rotation=0)
        axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cross_year_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-year performance plots saved to: {save_dir}/cross_year_performance_analysis.png")


def create_model_stability_analysis(regression_results, classification_results, save_dir="outputs/charts"):
    """
    Create model stability analysis visualization
    
    Args:
        regression_results (list): Regression results
        classification_results (list): Classification results
        save_dir (str): Directory to save plots
    """
    print("Creating model stability analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Regression stability
    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        stability_data = []
        
        for model in reg_df['Model'].unique():
            model_data = reg_df[reg_df['Model'] == model]
            mean_r2 = model_data['R2_Score'].mean()
            std_r2 = model_data['R2_Score'].std()
            stability_data.append({
                'Model': model,
                'Mean_R2': mean_r2,
                'Std_R2': std_r2,
                'Stability_Score': mean_r2 - std_r2  # Higher mean, lower std = more stable
            })
        
        stability_df = pd.DataFrame(stability_data).sort_values('Stability_Score', ascending=True)
        
        bars = axes[0].barh(stability_df['Model'], stability_df['Mean_R2'], 
                           xerr=stability_df['Std_R2'], capsize=5, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_title('Regression Model Stability (R² Score)')
        axes[0].set_xlabel('R² Score (Mean ± Std)')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add stability scores as text
        for i, (bar, score) in enumerate(zip(bars, stability_df['Stability_Score'])):
            axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'Stability: {score:.3f}', va='center', fontsize=9)
    
    # Classification stability
    if classification_results:
        class_df = pd.DataFrame(classification_results)
        stability_data = []
        
        for model in class_df['Model'].unique():
            model_data = class_df[class_df['Model'] == model]
            mean_acc = model_data['Accuracy'].mean()
            std_acc = model_data['Accuracy'].std()
            stability_data.append({
                'Model': model,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'Stability_Score': mean_acc - std_acc
            })
        
        stability_df = pd.DataFrame(stability_data).sort_values('Stability_Score', ascending=True)
        
        bars = axes[1].barh(stability_df['Model'], stability_df['Mean_Accuracy'], 
                           xerr=stability_df['Std_Accuracy'], capsize=5,
                           color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[1].set_title('Classification Model Stability (Accuracy)')
        axes[1].set_xlabel('Accuracy (Mean ± Std)')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add stability scores as text
        for i, (bar, score) in enumerate(zip(bars, stability_df['Stability_Score'])):
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'Stability: {score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_stability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model stability analysis saved to: {save_dir}/model_stability_analysis.png")


def create_professional_cross_year_tables(regression_results, classification_results, save_dir="outputs/tables"):
    """
    Create professional tables for cross-year analysis
    
    Args:
        regression_results (list): Regression results
        classification_results (list): Classification results
        save_dir (str): Directory to save tables
    """
    print("Creating professional cross-year tables...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Regression results table
    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        reg_df_rounded = reg_df.copy()
        reg_df_rounded['RMSE'] = reg_df_rounded['RMSE'].round(2)
        reg_df_rounded['MAE'] = reg_df_rounded['MAE'].round(2)
        reg_df_rounded['R2_Score'] = reg_df_rounded['R2_Score'].round(4)
        
        # Pivot table for better presentation
        reg_pivot = reg_df_rounded.pivot(index='Model', columns='Year', values='R2_Score')
        reg_pivot.columns = [f'{col} R²' for col in reg_pivot.columns]
        
        # Add summary statistics
        reg_pivot['Mean R²'] = reg_df_rounded.groupby('Model')['R2_Score'].mean().round(4)
        reg_pivot['Std R²'] = reg_df_rounded.groupby('Model')['R2_Score'].std().round(4)
        
        # Save in multiple formats
        reg_pivot.to_csv(f"{save_dir}/cross_year_regression_performance.csv")
        
        # Markdown format
        with open(f"{save_dir}/cross_year_regression_performance_markdown.txt", "w") as f:
            f.write("# Cross-Year Regression Model Performance\n\n")
            f.write(reg_pivot.to_markdown())
        
        # LaTeX format
        with open(f"{save_dir}/cross_year_regression_performance_latex.txt", "w") as f:
            f.write(reg_pivot.to_latex())
    
    # Classification results table
    if classification_results:
        class_df = pd.DataFrame(classification_results)
        class_df_rounded = class_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1_Score']:
            class_df_rounded[col] = class_df_rounded[col].round(4)
        
        # Pivot table for better presentation
        class_pivot = class_df_rounded.pivot(index='Model', columns='Year', values='Accuracy')
        class_pivot.columns = [f'{col} Acc' for col in class_pivot.columns]
        
        # Add summary statistics
        class_pivot['Mean Acc'] = class_df_rounded.groupby('Model')['Accuracy'].mean().round(4)
        class_pivot['Std Acc'] = class_df_rounded.groupby('Model')['Accuracy'].std().round(4)
        
        # Save in multiple formats
        class_pivot.to_csv(f"{save_dir}/cross_year_classification_performance.csv")
        
        # Markdown format
        with open(f"{save_dir}/cross_year_classification_performance_markdown.txt", "w") as f:
            f.write("# Cross-Year Classification Model Performance\n\n")
            f.write(class_pivot.to_markdown())
        
        # LaTeX format
        with open(f"{save_dir}/cross_year_classification_performance_latex.txt", "w") as f:
            f.write(class_pivot.to_latex())
    
    # Combined summary table
    summary_data = []
    
    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        for model in reg_df['Model'].unique():
            model_data = reg_df[reg_df['Model'] == model]
            summary_data.append({
                'Model': f"{model} (Regression)",
                'Task': 'Regression',
                'Mean_Performance': model_data['R2_Score'].mean(),
                'Std_Performance': model_data['R2_Score'].std(),
                'Best_Year': model_data.loc[model_data['R2_Score'].idxmax(), 'Year'],
                'Worst_Year': model_data.loc[model_data['R2_Score'].idxmin(), 'Year'],
                'Metric': 'R² Score'
            })
    
    if classification_results:
        class_df = pd.DataFrame(classification_results)
        for model in class_df['Model'].unique():
            model_data = class_df[class_df['Model'] == model]
            summary_data.append({
                'Model': f"{model} (Classification)",
                'Task': 'Classification',
                'Mean_Performance': model_data['Accuracy'].mean(),
                'Std_Performance': model_data['Accuracy'].std(),
                'Best_Year': model_data.loc[model_data['Accuracy'].idxmax(), 'Year'],
                'Worst_Year': model_data.loc[model_data['Accuracy'].idxmin(), 'Year'],
                'Metric': 'Accuracy'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df['Mean_Performance'] = summary_df['Mean_Performance'].round(4)
        summary_df['Std_Performance'] = summary_df['Std_Performance'].round(4)
        
        # Sort by performance
        summary_df = summary_df.sort_values('Mean_Performance', ascending=False)
        
        summary_df.to_csv(f"{save_dir}/cross_year_summary.csv", index=False)
        
        # Formatted table
        with open(f"{save_dir}/cross_year_summary_formatted.txt", "w") as f:
            f.write("CROSS-YEAR MODEL PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            col_widths = [25, 12, 8, 12, 12, 12]
            headers = ['Model', 'Mean Perf', 'Std', 'Best Year', 'Worst Year', 'Metric']
            
            # Header
            header_line = ""
            for header, width in zip(headers, col_widths):
                header_line += f"{header:<{width}}"
            f.write(header_line + "\n")
            f.write("-" * len(header_line) + "\n")
            
            # Data rows
            for _, row in summary_df.iterrows():
                data_line = ""
                data_line += f"{row['Model']:<{col_widths[0]}}"
                data_line += f"{row['Mean_Performance']:<{col_widths[1]}.4f}"
                data_line += f"{row['Std_Performance']:<{col_widths[2]}.4f}"
                data_line += f"{row['Best_Year']:<{col_widths[3]}}"
                data_line += f"{row['Worst_Year']:<{col_widths[4]}}"
                data_line += f"{row['Metric']:<{col_widths[5]}}"
                f.write(data_line + "\n")
    
    print("Professional cross-year tables saved:")
    print(f"  - Regression: {save_dir}/cross_year_regression_performance.*")
    print(f"  - Classification: {save_dir}/cross_year_classification_performance.*")
    print(f"  - Summary: {save_dir}/cross_year_summary.*")


def main():
    """
    Main cross-year analysis pipeline
    """
    print("="*60)
    print("CROSS-YEAR ANALYSIS PIPELINE")
    print("="*60)
    
    # Create output directories
    os.makedirs("outputs/charts", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    
    # Define dataset paths
    datasets = {
        '2015': 'data/2015-building-energy-benchmarking.csv',
        '2016': 'data/2016-building-energy-benchmarking.csv',
        '2021': 'data/energy_disclosure_2021_rows.csv'
    }
    
    # Load trained models
    regression_models, classification_models = load_trained_models()
    
    if not regression_models and not classification_models:
        print("❌ No trained models found. Please train models first.")
        return
    
    # Load feature names from unified dataset
    try:
        with open("outputs/feature_names.txt", "r") as f:
            unified_features = [line.strip() for line in f.readlines()]
        print(f"Unified model features: {len(unified_features)}")
    except FileNotFoundError:
        print("❌ Feature names not found. Please run multi_dataset_processor.py first.")
        return
    
    # Load scaler if available
    scaler = None
    try:
        scaler = joblib.load("outputs/scaler.pkl")
        print("✅ Scaler loaded")
    except FileNotFoundError:
        print("⚠️  Scaler not found, will proceed without scaling")
    
    # Test models on each year
    all_regression_results = []
    all_classification_results = []
    
    for year, file_path in datasets.items():
        print(f"\n" + "="*50)
        print(f"ANALYZING {year.upper()} DATASET")
        print("="*50)
        
        # Load and preprocess year data
        features, target, labels, info = load_individual_dataset(file_path, year)
        
        if features is None:
            print(f"❌ Failed to load {year} dataset")
            continue
        
        # Align features with unified model
        aligned_features = align_features_with_unified(features, unified_features)
        
        # Test regression models
        if regression_models:
            reg_results = test_regression_models_on_year(
                regression_models, aligned_features, target, year, scaler
            )
            all_regression_results.extend(reg_results)
        
        # Test classification models
        if classification_models:
            class_results = test_classification_models_on_year(
                classification_models, aligned_features, labels, year, scaler
            )
            all_classification_results.extend(class_results)
        
        print(f"✅ {year} analysis completed")
    
    # Create visualizations
    if all_regression_results or all_classification_results:
        create_cross_year_performance_plots(all_regression_results, all_classification_results)
        create_model_stability_analysis(all_regression_results, all_classification_results)
        
        # Create professional tables
        create_professional_cross_year_tables(all_regression_results, all_classification_results)
        
        # Print summary
        print(f"\n" + "="*60)
        print("CROSS-YEAR ANALYSIS COMPLETED")
        print("="*60)
        
        if all_regression_results:
            reg_df = pd.DataFrame(all_regression_results)
            print(f"\n📊 Regression Results Summary:")
            for model in reg_df['Model'].unique():
                model_data = reg_df[reg_df['Model'] == model]
                mean_r2 = model_data['R2_Score'].mean()
                std_r2 = model_data['R2_Score'].std()
                best_year = model_data.loc[model_data['R2_Score'].idxmax(), 'Year']
                print(f"  {model}: Mean R² = {mean_r2:.4f} (±{std_r2:.4f}), Best: {best_year}")
        
        if all_classification_results:
            class_df = pd.DataFrame(all_classification_results)
            print(f"\n🎯 Classification Results Summary:")
            for model in class_df['Model'].unique():
                model_data = class_df[class_df['Model'] == model]
                mean_acc = model_data['Accuracy'].mean()
                std_acc = model_data['Accuracy'].std()
                best_year = model_data.loc[model_data['Accuracy'].idxmax(), 'Year']
                print(f"  {model}: Mean Acc = {mean_acc:.4f} (±{std_acc:.4f}), Best: {best_year}")
        
        print(f"\n📁 Files Created:")
        print(f"  Charts: outputs/charts/cross_year_performance_analysis.png")
        print(f"  Charts: outputs/charts/model_stability_analysis.png")
        print(f"  Tables: outputs/tables/cross_year_*")
    
    else:
        print("❌ No results generated")
    
    return all_regression_results, all_classification_results


if __name__ == "__main__":
    main()