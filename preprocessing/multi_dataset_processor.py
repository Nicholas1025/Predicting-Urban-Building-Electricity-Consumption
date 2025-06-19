"""
Multi-Dataset Processor for Building Energy Consumption Analysis
Handles loading, cleaning, and merging multiple years of building energy data
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


def load_single_dataset(file_path, year=None):
    """
    Load a single dataset and add year information
    
    Args:
        file_path (str): Path to the CSV file
        year (str): Year identifier for the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset with year column
    """
    try:
        print(f"Loading dataset: {file_path}")
        df = pd.read_csv(file_path)
        
        # Add year column if provided
        if year:
            df['DataYear'] = year
        
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def standardize_column_names(df):
    """
    Standardize column names across different datasets
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with standardized column names
    """
    # Common column name mappings
    column_mappings = {
        # Energy consumption columns
        'SiteEnergyUse(kBtu)': 'SiteEnergyUse',
        'SiteEnergyUseWN(kBtu)': 'SiteEnergyUse',
        'SiteEUI(kBtu/sf)': 'SiteEUI',
        'SiteEUIWN(kBtu/sf)': 'SiteEUI',
        
        # Building information
        'PropertyName': 'BuildingName',
        'PrimaryPropertyType': 'PropertyType',
        'PropertyGFATotal': 'TotalGrossFloorArea',
        'PropertyGFAParking': 'ParkingGrossFloorArea',
        'PropertyGFABuilding(s)': 'BuildingGrossFloorArea',
        
        # Energy types
        'Electricity(kWh)': 'ElectricityUse',
        'NaturalGas(therms)': 'NaturalGasUse',
        'Steam(kBtu)': 'SteamUse',
        
        # Year information
        'YearBuilt': 'YearBuilt',
        'NumberofBuildings': 'NumberOfBuildings',
        'NumberofFloors': 'NumberOfFloors'
    }
    
    # Apply mappings
    df_renamed = df.rename(columns=column_mappings)
    
    # Standardize column names (remove special characters, make consistent)
    new_columns = []
    for col in df_renamed.columns:
        # Remove special characters and spaces
        new_col = col.replace('(', '').replace(')', '').replace('/', '_').replace(' ', '')
        new_columns.append(new_col)
    
    df_renamed.columns = new_columns
    
    return df_renamed


def align_datasets_schema(datasets):
    """
    Align multiple datasets to have consistent schema
    
    Args:
        datasets (list): List of DataFrames
        
    Returns:
        list: List of aligned DataFrames
    """
    if not datasets:
        return []
    
    print("Aligning dataset schemas...")
    
    # Find common columns across all datasets
    common_columns = set(datasets[0].columns)
    for df in datasets[1:]:
        common_columns = common_columns.intersection(set(df.columns))
    
    print(f"Common columns found: {len(common_columns)}")
    
    # Find all unique columns
    all_columns = set()
    for df in datasets:
        all_columns = all_columns.union(set(df.columns))
    
    missing_columns_info = {}
    for i, df in enumerate(datasets):
        missing = all_columns - set(df.columns)
        if missing:
            missing_columns_info[f'Dataset_{i+1}'] = missing
    
    if missing_columns_info:
        print("Missing columns by dataset:")
        for dataset, missing in missing_columns_info.items():
            print(f"  {dataset}: {list(missing)[:5]}..." if len(missing) > 5 else f"  {dataset}: {list(missing)}")
    
    # Align datasets - keep only common columns for now
    aligned_datasets = []
    common_columns_list = sorted(list(common_columns))
    
    for i, df in enumerate(datasets):
        aligned_df = df[common_columns_list].copy()
        print(f"Dataset {i+1} aligned shape: {aligned_df.shape}")
        aligned_datasets.append(aligned_df)
    
    return aligned_datasets


def create_unified_target_variable(df, target_candidates=['SiteEnergyUse', 'SiteEnergyUse_kBtu']):
    """
    Create a unified target variable from available energy consumption columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_candidates (list): List of potential target column names
        
    Returns:
        tuple: (dataframe with target, target_column_name)
    """
    target_col = None
    
    # Find the best target column
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break
    
    # If no direct match, look for similar columns
    if target_col is None:
        for col in df.columns:
            if 'energy' in col.lower() and 'site' in col.lower():
                target_col = col
                break
    
    if target_col is None:
        # Use first numerical column as fallback
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            target_col = numerical_cols[0]
            print(f"Warning: Using {target_col} as target variable (no standard energy column found)")
    
    # Create standardized target column
    if target_col:
        df['EnergyConsumption'] = df[target_col]
        print(f"Target variable created from: {target_col}")
        return df, 'EnergyConsumption'
    else:
        raise ValueError("No suitable target variable found in dataset")


def create_energy_efficiency_labels(energy_values, method='quartiles'):
    """
    Create energy efficiency classification labels from continuous energy values
    
    Args:
        energy_values (pd.Series): Energy consumption values
        method (str): Method for creating labels ('quartiles', 'percentiles')
        
    Returns:
        pd.Series: Classification labels
    """
    if method == 'quartiles':
        # 4-class classification based on quartiles
        q1 = energy_values.quantile(0.25)
        q2 = energy_values.quantile(0.50)
        q3 = energy_values.quantile(0.75)
        
        def assign_efficiency_label(value):
            if pd.isna(value):
                return 'Unknown'
            elif value <= q1:
                return 'Excellent'  # Low energy use = High efficiency
            elif value <= q2:
                return 'Good'
            elif value <= q3:
                return 'Average'
            else:
                return 'Poor'  # High energy use = Low efficiency
        
        labels = energy_values.apply(assign_efficiency_label)
        
    elif method == 'percentiles':
        # 5-class classification based on percentiles
        p20 = energy_values.quantile(0.20)
        p40 = energy_values.quantile(0.40)
        p60 = energy_values.quantile(0.60)
        p80 = energy_values.quantile(0.80)
        
        def assign_efficiency_label(value):
            if pd.isna(value):
                return 'Unknown'
            elif value <= p20:
                return 'A+'  # Best efficiency
            elif value <= p40:
                return 'A'
            elif value <= p60:
                return 'B'
            elif value <= p80:
                return 'C'
            else:
                return 'D'  # Worst efficiency
        
        labels = energy_values.apply(assign_efficiency_label)
    
    print(f"Energy efficiency labels distribution:")
    print(labels.value_counts())
    
    return labels


def clean_and_prepare_unified_dataset(df):
    """
    Clean and prepare the unified dataset for machine learning
    
    Args:
        df (pd.DataFrame): Combined dataset
        
    Returns:
        tuple: (features_df, target_series, classification_labels)
    """
    print("Cleaning and preparing unified dataset...")
    
    # Create target variable
    df, target_col = create_unified_target_variable(df)
    
    # Remove rows with invalid target values
    valid_mask = ~(df[target_col].isnull() | (df[target_col] <= 0))
    df_clean = df[valid_mask].copy()
    
    print(f"Removed {len(df) - len(df_clean)} rows with invalid target values")
    print(f"Remaining samples: {len(df_clean)}")
    
    # Create classification labels
    classification_labels = create_energy_efficiency_labels(df_clean[target_col])
    
    # Separate features and target
    target_series = df_clean[target_col].copy()
    features_df = df_clean.drop(columns=[target_col]).copy()
    
    # Drop irrelevant columns
    columns_to_drop = [
        'OSEBuildingID', 'BuildingName', 'Address', 'City', 'State',
        'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 
        'TaxParcelIdentificationNumber', 'ComplianceStatus', 
        'Comments', 'DefaultData', 'ListOfAllPropertyUseTypes'
    ]
    
    existing_cols_to_drop = [col for col in columns_to_drop if col in features_df.columns]
    if existing_cols_to_drop:
        features_df = features_df.drop(columns=existing_cols_to_drop)
        print(f"Dropped {len(existing_cols_to_drop)} irrelevant columns")
    
    # Handle categorical variables
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if features_df[col].nunique() > 50:  # Too many unique values
            print(f"Dropping {col} - too many unique values ({features_df[col].nunique()})")
            features_df = features_df.drop(columns=[col])
        else:
            # Label encode categorical variables
            le = LabelEncoder()
            features_df[col] = features_df[col].fillna('Unknown')
            features_df[col] = le.fit_transform(features_df[col].astype(str))
    
    # Handle missing values in numerical columns
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        features_df[numerical_cols] = features_df[numerical_cols].fillna(features_df[numerical_cols].median())
    
    print(f"Final dataset shape: {features_df.shape}")
    print(f"Final features: {len(features_df.columns)}")
    
    return features_df, target_series, classification_labels


def load_and_merge_datasets(data_paths):
    """
    Load and merge multiple building energy datasets
    
    Args:
        data_paths (dict): Dictionary with year as key and file path as value
        
    Returns:
        tuple: (unified_features, unified_target, unified_labels, dataset_info)
    """
    print("="*60)
    print("MULTI-DATASET LOADING AND PROCESSING")
    print("="*60)
    
    # Load individual datasets
    datasets = []
    for year, path in data_paths.items():
        df = load_single_dataset(path, year)
        if df is not None:
            # Standardize column names
            df_std = standardize_column_names(df)
            datasets.append(df_std)
    
    if not datasets:
        raise ValueError("No datasets could be loaded successfully")
    
    print(f"\nSuccessfully loaded {len(datasets)} datasets")
    
    # Align dataset schemas
    aligned_datasets = align_datasets_schema(datasets)
    
    # Merge datasets
    print("\nMerging datasets...")
    unified_df = pd.concat(aligned_datasets, ignore_index=True)
    print(f"Unified dataset shape: {unified_df.shape}")
    
    # Clean and prepare
    features, target, labels = clean_and_prepare_unified_dataset(unified_df)
    
    # Create dataset info
    dataset_info = {
        'total_samples': len(features),
        'total_features': len(features.columns),
        'datasets_used': list(data_paths.keys()),
        'target_variable': 'EnergyConsumption',
        'classification_classes': labels.value_counts().to_dict()
    }
    
    print("\n" + "="*60)
    print("DATASET PROCESSING COMPLETED")
    print("="*60)
    print(f"Total samples: {dataset_info['total_samples']}")
    print(f"Total features: {dataset_info['total_features']}")
    print(f"Datasets used: {dataset_info['datasets_used']}")
    print(f"Classification distribution: {dataset_info['classification_classes']}")
    
    return features, target, labels, dataset_info


def save_processed_datasets(features, target, labels, output_dir="outputs"):
    """
    Save processed datasets for model training
    
    Args:
        features (pd.DataFrame): Feature matrix
        target (pd.Series): Target variable
        labels (pd.Series): Classification labels
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving processed datasets to {output_dir}/...")
    
    # Save unified dataset
    features.to_csv(f"{output_dir}/unified_features.csv", index=False)
    target.to_csv(f"{output_dir}/unified_target.csv", index=False)
    labels.to_csv(f"{output_dir}/unified_labels.csv", index=False)
    
    # Save feature names
    with open(f"{output_dir}/feature_names.txt", "w") as f:
        for name in features.columns:
            f.write(f"{name}\n")
    
    print("✅ Datasets saved successfully:")
    print(f"  - unified_features.csv ({features.shape})")
    print(f"  - unified_target.csv ({len(target)} samples)")
    print(f"  - unified_labels.csv ({len(labels)} samples)")
    print(f"  - feature_names.txt ({len(features.columns)} features)")


def main():
    """
    Main processing pipeline
    """
    # Define dataset paths
    data_paths = {
        '2015': 'data/2015-building-energy-benchmarking.csv',
        '2016': 'data/2016-building-energy-benchmarking.csv', 
        '2021': 'data/energy_disclosure_2021_rows.csv'
    }
    
    try:
        # Load and process datasets
        features, target, labels, info = load_and_merge_datasets(data_paths)
        
        # Save processed data
        save_processed_datasets(features, target, labels)
        
        print("\n🎉 Multi-dataset processing completed successfully!")
        return features, target, labels, info
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return None


if __name__ == "__main__":
    main()