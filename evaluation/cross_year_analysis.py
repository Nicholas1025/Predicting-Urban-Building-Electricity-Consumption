"""
Cross-Year Analysis for Building Energy Consumption Models (FIXED VERSION)
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
    Load and preprocess individual year dataset with improved error handling
    """
    print(f"Loading {year} dataset from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"  Raw shape: {df.shape}")
        
        # Apply same preprocessing as unified dataset
        from preprocessing.multi_dataset_processor import (
            standardize_column_names, create_energy_efficiency_labels
        )
        
        # Standardize columns
        df_std = standardize_column_names(df)
        
        # Create target variable with better error handling
        target_col = None
        target_candidates = [
            'SiteEnergyUse', 'SiteEnergyUse_kBtu', 'SiteEnergyUseWN_kBtu',
            'TotalGHGEmissions', 'ENERGYSTARScore'
        ]
        
        # Find suitable target column
        for candidate in target_candidates:
            if candidate in df_std.columns and df_std[candidate].notna().sum() > 50:
                target_col = candidate
                break
        
        # If no energy column found, use first numerical column
        if target_col is None:
            numerical_cols = df_std.select_dtypes(include=[np.number]).columns
            non_id_cols = [col for col in numerical_cols if 'ID' not in col.upper() and 'BBL' not in col.upper()]
            if len(non_id_cols) > 0:
                target_col = non_id_cols[0]
                print(f"Warning: Using {target_col} as target variable")
            else:
                print(f"No suitable target variable found in {year} dataset")
                return None, None, None, None
        
        # Create unified target column
        df_with_target = df_std.copy()
        df_with_target['EnergyConsumption'] = df_with_target[target_col]
        target_col = 'EnergyConsumption'
        
        print(f"Target variable created from: {target_col}")
        
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
        
        # Drop irrelevant columns
        columns_to_drop = [
            'OSEBuildingID', 'BuildingName', 'Address', 'City', 'State',
            'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 
            'TaxParcelIdentificationNumber', 'ComplianceStatus', 
            'Comments', 'DefaultData', 'ListOfAllPropertyUseTypes',
            'EnergyConsumption', '10_Digit_BBL', 'Street_Number', 'Street_Name'
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
                    print(f"Warning: Could not encode {col}, dropping it")
                    features = features.drop(columns=[col])
        
        # Handle missing values
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            for col in numerical_cols:
                if features[col].isnull().sum() > 0:
                    median_val = features[col].median()
                    if pd.isna(median_val):
                        features[col] = features[col].fillna(0)
                    else:
                        features[col] = features[col].fillna(median_val)
        
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
        return None, None, None, None


def align_features_with_unified(features, unified_feature_names):
    """
    Align individual dataset features with unified model features
    """
    # Find common features
    common_features = [col for col in unified_feature_names if col in features.columns]
    missing_features = [col for col in unified_feature_names if col not in features.columns]
    
    if missing_features:
        print(f"  Missing features: {len(missing_features)} (will be filled with zeros)")
    
    # Create a new DataFrame with all required features
    aligned_features = pd.DataFrame(index=features.index)
    
    # Add common features
    for feature in common_features:
        aligned_features[feature] = features[feature]
    
    # Add missing features with zero values
    for feature in missing_features:
        aligned_features[feature] = 0
    
    # Ensure column order matches unified model
    aligned_features = aligned_features[unified_feature_names]
    
    print(f"  Features aligned: {aligned_features.shape}")
    
    return aligned_features


def load_trained_models():
    """Load trained regression and classification models"""
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
            if os.path.exists(file_path):
                model = joblib.load(file_path)
                regression_models[model_name] = model
                print(f"  Loaded {model_name} regression model")
            else:
                print(f"  {model_name} regression model not found")
        except Exception as e:
            print(f"  Error loading {model_name} regression model: {e}")
    
    # Load classification models
    for model_name, file_path in classification_files.items():
        try:
            if os.path.exists(file_path):
                model = joblib.load(file_path)
                classification_models[model_name] = model
                print(f"  Loaded {model_name} classification model")
            else:
                print(f"  {model_name} classification model not found")
        except Exception as e:
            print(f"  Error loading {model_name} classification model: {e}")
    
    return regression_models, classification_models


def test_regression_models_on_year(models, features, target, year, scaler=None):
    """Test regression models on specific year data"""
    print(f"\nTesting regression models on {year} data...")
    
    results = []
    
    # Scale features if scaler is available
    if scaler is not None:
        try:
            features_scaled = pd.DataFrame(
                scaler.transform(features), 
                columns=features.columns, 
                index=features.index
            )
        except Exception as e:
            print(f"  Warning: Scaling failed, using unscaled features")
            features_scaled = features
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
            
            print(f"  {model_name}: R2 = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
    
    return results


def create_cross_year_performance_plots(regression_results, classification_results, save_dir="outputs/charts"):
    """Create cross-year performance visualization plots"""
    print("\nCreating cross-year performance plots...")
    
    # Convert to DataFrames
    reg_df = pd.DataFrame(regression_results) if regression_results else pd.DataFrame()
    class_df = pd.DataFrame(classification_results) if classification_results else pd.DataFrame()
    
    if len(reg_df) == 0 and len(class_df) == 0:
        print("No results to plot")
        return
    
    # Create a simple plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if len(reg_df) > 0:
        # Plot regression results
        for model in reg_df['Model'].unique():
            model_data = reg_df[reg_df['Model'] == model]
            ax.plot(model_data['Year'], model_data['R2_Score'], marker='o', label=f'{model} R2')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('R2 Score')
        ax.set_title('Cross-Year Model Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No regression results available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cross_year_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-year performance plots saved to: {save_dir}/cross_year_performance_analysis.png")


def create_professional_cross_year_tables(regression_results, classification_results, save_dir="outputs/tables"):
    """Create professional tables for cross-year analysis"""
    print("Creating professional cross-year tables...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
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
                    'Std_Performance': model_data['R2_Score'].std() if len(model_data) > 1 else 0,
                    'Best_Year': model_data.loc[model_data['R2_Score'].idxmax(), 'Year'],
                    'Metric': 'R2 Score'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df['Mean_Performance'] = summary_df['Mean_Performance'].round(4)
            summary_df['Std_Performance'] = summary_df['Std_Performance'].round(4)
            
            summary_df = summary_df.sort_values('Mean_Performance', ascending=False)
            summary_df.to_csv(f"{save_dir}/cross_year_summary.csv", index=False)
            print(f"Summary table saved to: {save_dir}/cross_year_summary.csv")
    
    except Exception as e:
        print(f"Error creating tables: {e}")


def main():
    """Main cross-year analysis pipeline"""
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
    
    if not regression_models:
        print("No trained regression models found. Skipping cross-year analysis.")
        return [], []
    
    # Load feature names from unified dataset
    try:
        with open("outputs/feature_names.txt", "r") as f:
            unified_features = [line.strip() for line in f.readlines()]
        print(f"Unified model features: {len(unified_features)}")
    except FileNotFoundError:
        print("Feature names not found. Please run training first.")
        return [], []
    
    # Load scaler if available
    scaler = None
    try:
        scaler = joblib.load("outputs/scaler.pkl")
        print("Scaler loaded")
    except FileNotFoundError:
        print("Scaler not found, will proceed without scaling")
    
    # Test models on each year
    all_regression_results = []
    
    for year, file_path in datasets.items():
        print(f"\n" + "="*50)
        print(f"ANALYZING {year.upper()} DATASET")
        print("="*50)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        # Load and preprocess year data
        features, target, labels, info = load_individual_dataset(file_path, year)
        
        if features is None:
            print(f"Failed to load {year} dataset")
            continue
        
        # Align features with unified model
        try:
            aligned_features = align_features_with_unified(features, unified_features)
        except Exception as e:
            print(f"Failed to align features for {year}: {e}")
            continue
        
        # Test regression models
        if regression_models:
            reg_results = test_regression_models_on_year(
                regression_models, aligned_features, target, year, scaler
            )
            all_regression_results.extend(reg_results)
        
        print(f"Analysis completed for {year}")
    
    # Create visualizations and tables if we have results
    if all_regression_results:
        try:
            create_cross_year_performance_plots(all_regression_results, [])
        except Exception as e:
            print(f"Could not create performance plots: {e}")
        
        try:
            create_professional_cross_year_tables(all_regression_results, [])
        except Exception as e:
            print(f"Could not create tables: {e}")
        
        # Print summary
        print(f"\n" + "="*60)
        print("CROSS-YEAR ANALYSIS COMPLETED")
        print("="*60)
        
        reg_df = pd.DataFrame(all_regression_results)
        print(f"\nRegression Results Summary:")
        for model in reg_df['Model'].unique():
            model_data = reg_df[reg_df['Model'] == model]
            mean_r2 = model_data['R2_Score'].mean()
            best_year = model_data.loc[model_data['R2_Score'].idxmax(), 'Year']
            print(f"  {model}: Mean R2 = {mean_r2:.4f}, Best: {best_year}")
        
        print(f"\nFiles Created:")
        print(f"  Charts: outputs/charts/cross_year_performance_analysis.png")
        print(f"  Tables: outputs/tables/cross_year_summary.csv")
    
    else:
        print("No results generated")
    
    return all_regression_results, []


if __name__ == "__main__":
    main()
