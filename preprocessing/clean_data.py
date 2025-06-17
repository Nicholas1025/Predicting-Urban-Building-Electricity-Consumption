"""
Data Preprocessing Script for Seattle Building Energy Dataset
Loads, cleans, and prepares data for machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load the Seattle building energy dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def drop_irrelevant_columns(df):
    """
    Drop columns that are not useful for prediction
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with irrelevant columns removed
    """
    # Columns to drop (IDs, addresses, and other non-predictive features)
    columns_to_drop = [
        'OSEBuildingID', 'DataYear', 'BuildingName', 'Address', 'City', 'State',
        'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 'TaxParcelIdentificationNumber',
        'ComplianceStatus', 'Comments', 'DefaultData', 'ListOfAllPropertyUseTypes'
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_columns_to_drop)
    
    print(f"Dropped {len(existing_columns_to_drop)} irrelevant columns")
    print(f"Remaining columns: {df_cleaned.shape[1]}")
    
    return df_cleaned


def handle_target_variable(df, target_col='SiteEnergyUse(kBtu)'):
    """
    Extract and clean the target variable
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        
    Returns:
        tuple: (features_df, target_series)
    """
    if target_col not in df.columns:
        # Try alternative column names
        alternative_names = ['SiteEnergyUseWN(kBtu)', 'SiteEnergyUse', 'TotalGHGEmissions']
        for alt_name in alternative_names:
            if alt_name in df.columns:
                target_col = alt_name
                break
        else:
            print("Warning: Target column not found. Using first numerical column.")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            target_col = numerical_cols[0] if len(numerical_cols) > 0 else df.columns[-1]
    
    print(f"Using '{target_col}' as target variable")
    
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Remove rows where target is null or invalid
    valid_indices = ~(y.isnull() | (y <= 0))  # Remove null and non-positive values
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Removed {len(df) - len(X)} rows with invalid target values")
    print(f"Final dataset shape: {X.shape}")
    
    return X, y


def encode_categorical_features(X):
    """
    Encode categorical features using Label Encoding
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    X_encoded = X.copy()
    categorical_columns = X_encoded.select_dtypes(include=['object']).columns
    
    print(f"Encoding {len(categorical_columns)} categorical columns")
    
    for col in categorical_columns:
        if X_encoded[col].nunique() > 100:  # Skip columns with too many unique values
            print(f"Dropping {col} - too many unique values ({X_encoded[col].nunique()})")
            X_encoded = X_encoded.drop(columns=[col])
        else:
            le = LabelEncoder()
            # Handle missing values by treating them as a separate category
            X_encoded[col] = X_encoded[col].fillna('Unknown')
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            print(f"Encoded {col} - {X_encoded[col].nunique()} unique values")
    
    return X_encoded


def handle_missing_values(X):
    """
    Handle missing values in numerical features
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        pd.DataFrame: Dataframe with imputed missing values
    """
    print("Handling missing values...")
    
    # Separate numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 0:
        # Impute numerical columns with median
        numerical_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
        print(f"Imputed missing values in {len(numerical_cols)} numerical columns")
    
    return X


def feature_selection_and_cleaning(X):
    """
    Additional feature cleaning and selection
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        pd.DataFrame: Cleaned features dataframe
    """
    print("Performing feature selection and cleaning...")
    
    # Remove columns with too many missing values (>50%)
    missing_threshold = 0.5
    missing_percentages = X.isnull().sum() / len(X)
    columns_to_remove = missing_percentages[missing_percentages > missing_threshold].index
    
    if len(columns_to_remove) > 0:
        X = X.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} columns with >50% missing values")
    
    # Remove constant columns
    constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_columns:
        X = X.drop(columns=constant_columns)
        print(f"Removed {len(constant_columns)} constant columns")
    
    # Convert boolean columns to integers
    boolean_cols = X.select_dtypes(include=['bool']).columns
    if len(boolean_cols) > 0:
        X[boolean_cols] = X[boolean_cols].astype(int)
        print(f"Converted {len(boolean_cols)} boolean columns to integers")
    
    return X


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and apply feature scaling
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print(f"Splitting data into train ({1-test_size:.0%}) and test ({test_size:.0%}) sets...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Target variable range - Train: [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"Target variable range - Test: [{y_test.min():.0f}, {y_test.max():.0f}]")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def preprocess_data(file_path, target_col='SiteEnergyUse(kBtu)', test_size=0.2, random_state=42):
    """
    Complete data preprocessing pipeline
    
    Args:
        file_path (str): Path to the CSV file
        target_col (str): Name of target column
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print("="*50)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*50)
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    print(f"\nInitial dataset info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Drop irrelevant columns
    df = drop_irrelevant_columns(df)
    
    # Handle target variable
    X, y = handle_target_variable(df, target_col)
    
    # Encode categorical features
    X = encode_categorical_features(X)
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Additional cleaning
    X = feature_selection_and_cleaning(X)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size, random_state)
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, scaler, X_train.columns.tolist()


if __name__ == "__main__":
    # Main execution
    file_path = "data/2016-building-energy-benchmarking.csv"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    
    # Run preprocessing
    result = preprocess_data(file_path)
    
    if result is not None:
        X_train, X_test, y_train, y_test, scaler, feature_names = result
        
        # Save preprocessed data for other scripts to use
        print("\nSaving preprocessed data...")
        X_train.to_csv("outputs/X_train.csv", index=False)
        X_test.to_csv("outputs/X_test.csv", index=False)
        y_train.to_csv("outputs/y_train.csv", index=False)
        y_test.to_csv("outputs/y_test.csv", index=False)
        
        # Save feature names
        with open("outputs/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        print("Preprocessed data saved to outputs/ directory")
        print("\nSummary statistics:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train stats: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
    else:
        print("Preprocessing failed. Please check the input file and try again.")