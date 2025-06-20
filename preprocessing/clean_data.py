"""
Enhanced Data Preprocessing Script for Multiple Building Energy Datasets
FIXED: Complete version with proper find_target_variable_enhanced function
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
import json
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load building energy dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {os.path.basename(file_path)}")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def identify_dataset_type(df, file_path):
    """
    Identify which dataset type based on columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        file_path (str): Path to identify dataset
        
    Returns:
        str: Dataset type ('seattle_2016', 'seattle_2015', 'nyc_2021')
    """
    filename = os.path.basename(file_path).lower()
    
    if '2016' in filename or 'Address' in df.columns:
        return 'seattle_2016'
    elif '2015' in filename or 'Location' in df.columns:
        return 'seattle_2015'
    elif '2021' in filename or '10_Digit_BBL' in df.columns:
        return 'nyc_2021'
    else:
        # Try to infer from columns
        if 'Address' in df.columns and 'City' in df.columns:
            return 'seattle_2016'
        elif 'Location' in df.columns:
            return 'seattle_2015'
        elif '10_Digit_BBL' in df.columns:
            return 'nyc_2021'
        else:
            print("Warning: Could not identify dataset type, assuming Seattle 2016 format")
            return 'seattle_2016'


def find_target_variable_enhanced(df, dataset_type=None):
    """
    Enhanced target variable detection with NYC 2021 specific handling
    FIXED VERSION - 确保对所有数据集都能正确工作
    """
    print("Enhanced target variable detection...")
    print(f"Dataset type: {dataset_type}")
    print(f"Available columns: {len(df.columns)}")
    
    if 'Energy_Star_1-100_Score' in df.columns:
        valid_count = df['Energy_Star_1-100_Score'].notna().sum()
        if valid_count > 10:  # 降低阈值适应小数据集
            print(f"✓ Found NYC 2021 target: Energy_Star_1-100_Score (with {valid_count} valid values)")
            return 'Energy_Star_1-100_Score'
    
    if dataset_type == 'nyc_2021':
        min_valid_threshold = 10  
        energy_patterns = [
            'Energy_Star_1-100_Score',
            'DOF_Gross_Square_Footage',  
        ]
    else:
        min_valid_threshold = 50  # Seattle 数据集较大
        energy_patterns = [
            'SiteEnergyUse', 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)', 
            'SourceEUI(kBtu/sf)', 'SiteEUI(kBtu/sf)',
            'TotalGHGEmissions', 'ENERGYSTARScore',
            'site_energy_use', 'source_energy_use',
        ]
    
    print(f"Using minimum valid threshold: {min_valid_threshold}")

    for pattern in energy_patterns:
        if pattern in df.columns:
            valid_count = df[pattern].notna().sum()
            if valid_count >= min_valid_threshold:
                print(f"✓ Found target variable: {pattern} (with {valid_count} valid values)")
                return pattern

    df_columns_lower = [col.lower() for col in df.columns]
    for pattern in energy_patterns:
        pattern_lower = pattern.lower()
        for i, col_lower in enumerate(df_columns_lower):
            if pattern_lower == col_lower:
                actual_col = df.columns[i]
                valid_count = df[actual_col].notna().sum()
                if valid_count >= min_valid_threshold:
                    print(f"✓ Found target variable (case-insensitive): {actual_col}")
                    return actual_col

    energy_keywords = ['energy', 'eui', 'kbtu', 'ghg', 'emission', 'star', 'consumption']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in energy_keywords):
            if df[col].dtype in ['int64', 'float64']:
                valid_count = df[col].notna().sum()
                if valid_count >= min_valid_threshold:
                    try:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
                            print(f"✓ Found potential target (partial): {col}")
                            return col
                    except:
                        continue

    print("Looking for any suitable numerical column...")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_keywords = ['id', 'bbl', 'bin', 'lat', 'lon', 'x_coord', 'y_coord', 'zip', 'year']
    
    for col in numerical_cols:
        col_lower = col.lower()
        if not any(keyword in col_lower for keyword in exclude_keywords):
            valid_count = df[col].notna().sum()
            if valid_count >= min_valid_threshold:
                try:
                    variance = df[col].var()
                    if pd.notna(variance) and variance > 0:
                        print(f"✓ Found fallback target: {col}")
                        return col
                except:
                    continue
    
    print("❌ No suitable target variable found!")
    return None


def preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Enhanced data preprocessing function for individual datasets with improved target detection
    
    Args:
        file_path (str): CSV file path
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print(f"Loading data from: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Identify dataset type
    dataset_type = identify_dataset_type(df, file_path)
    print(f"Dataset type: {dataset_type}")
    
    # Enhanced target variable detection
    target_col = find_target_variable_enhanced(df, dataset_type)
    
    if target_col is None:
        print("ERROR: No suitable target variable found!")
        print("Available columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1:2d}. {col} ({df[col].dtype})")
        raise ValueError("No suitable target variable found")
    
    print(f"Using '{target_col}' as target variable")
    
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Remove invalid target values
    if 'energy' in target_col.lower() or 'ghg' in target_col.lower() or 'emission' in target_col.lower():
        valid_mask = ~(y.isnull() | (y <= 0))
    elif 'star' in target_col.lower() or 'score' in target_col.lower():
        # For scores, keep values between 1-100
        valid_mask = ~(y.isnull()) & (y >= 1) & (y <= 100)
    else:
        # For other metrics, just remove null values
        valid_mask = ~y.isnull()
    
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"After removing invalid targets: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Delete irrelevant columns
    irrelevant_cols = [
        'OSEBuildingID', 'BuildingName', 'PropertyName', 'Address', 'City', 'State',
        'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 'TaxParcelIdentificationNumber',
        'ComplianceStatus', 'Comments', 'DefaultData', 'ListOfAllPropertyUseTypes',
        '10_Digit_BBL', 'Street_Number', 'Street_Name',
        # NYC specific columns to drop
        'Property_Name', 'Primary_Property_Type___Self_Selected',
        'Borough', 'Postcode', 'BBL___10_digits',
        'NYC_Borough__Building_Class___Tax_Class_1',
        'NYC_Building_Identification_Number__BIN_',
        'Reported_Address', 'Property_Floor_Area_Buildings_sq_ft'
    ]
    
    existing_irrelevant = [col for col in irrelevant_cols if col in X.columns]
    if existing_irrelevant:
        X = X.drop(columns=existing_irrelevant)
        print(f"Dropped {len(existing_irrelevant)} irrelevant columns")
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if X[col].nunique() > 50:  # Too many unique values
            X = X.drop(columns=[col])
            print(f"Dropped {col} - too many unique values")
        else:
            # Label encoding
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            try:
                X[col] = le.fit_transform(X[col].astype(str))
            except Exception as e:
                print(f"Warning: Could not encode {col}, dropping it")
                X = X.drop(columns=[col])
    
    # Handle missing values
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_val)
    
    # Remove constant columns
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"Dropped {len(constant_cols)} constant columns")
    
    # Remove highly correlated features
    if len(X.columns) > 1:
        try:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [column for column in upper_triangle.columns 
                                 if any(upper_triangle[column] > 0.95)]
            
            if high_corr_features:
                X = X.drop(columns=high_corr_features)
                print(f"Dropped {len(high_corr_features)} highly correlated features")
        except Exception as e:
            print(f"Warning: Could not perform correlation analysis: {e}")
    
    print(f"Final feature count: {X.shape[1]}")
    
    if X.shape[1] == 0:
        raise ValueError("No features remaining after preprocessing")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    feature_names = X_train.columns.tolist()
    
    print(f"Train set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print(f"Target range: [{y.min():.0f}, {y.max():.0f}]")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# Keep all other existing functions unchanged...
def standardize_seattle_2015(df):
    """
    Standardize Seattle 2015 dataset to common format
    
    Args:
        df (pd.DataFrame): Seattle 2015 dataframe
        
    Returns:
        pd.DataFrame: Standardized dataframe
    """
    df_std = df.copy()
    
    # Parse Location column to extract Address, City, State, ZipCode
    if 'Location' in df_std.columns:
        print("Parsing Location column for Seattle 2015 data...")
        
        def parse_location(location_str):
            """Parse the location JSON string"""
            try:
                if pd.isna(location_str):
                    return {'address': None, 'city': None, 'state': None, 'zip': None, 
                           'latitude': None, 'longitude': None}
                
                # Handle the nested JSON structure
                if isinstance(location_str, str):
                    # Extract latitude and longitude
                    import re
                    lat_match = re.search(r"'latitude': '([^']*)'", location_str)
                    lon_match = re.search(r"'longitude': '([^']*)'", location_str)
                    
                    latitude = float(lat_match.group(1)) if lat_match else None
                    longitude = float(lon_match.group(1)) if lon_match else None
                    
                    # Extract human_address JSON
                    addr_match = re.search(r"'human_address': '({[^}]*})'", location_str)
                    if addr_match:
                        addr_json = addr_match.group(1).replace('\\"', '"')
                        try:
                            addr_data = json.loads(addr_json)
                            return {
                                'address': addr_data.get('address'),
                                'city': addr_data.get('city'),
                                'state': addr_data.get('state'),
                                'zip': addr_data.get('zip'),
                                'latitude': latitude,
                                'longitude': longitude
                            }
                        except:
                            pass
                
                return {'address': None, 'city': None, 'state': None, 'zip': None,
                       'latitude': None, 'longitude': None}
                       
            except Exception as e:
                return {'address': None, 'city': None, 'state': None, 'zip': None,
                       'latitude': None, 'longitude': None}
        
        # Apply parsing
        location_data = df_std['Location'].apply(parse_location)
        
        # Add parsed columns
        df_std['Address'] = [item['address'] for item in location_data]
        df_std['City'] = [item['city'] for item in location_data]
        df_std['State'] = [item['state'] for item in location_data]
        df_std['ZipCode'] = [item['zip'] for item in location_data]
        if 'Latitude' not in df_std.columns:
            df_std['Latitude'] = [item['latitude'] for item in location_data]
        if 'Longitude' not in df_std.columns:
            df_std['Longitude'] = [item['longitude'] for item in location_data]
        
        # Drop original Location column
        df_std = df_std.drop(columns=['Location'])
    
    # Rename columns to match 2016 format
    column_mapping = {
        'GHGEmissions(MetricTonsCO2e)': 'TotalGHGEmissions',
        'GHGEmissionsIntensity(kgCO2e/ft2)': 'GHGEmissionsIntensity',
        'Comment': 'Comments'
    }
    
    df_std = df_std.rename(columns=column_mapping)
    
    # Add missing columns with default values
    if 'Outlier' not in df_std.columns:
        df_std['Outlier'] = False
    
    return df_std


def standardize_nyc_2021(df):
    """
    Standardize NYC 2021 dataset to common format
    
    Args:
        df (pd.DataFrame): NYC 2021 dataframe
        
    Returns:
        pd.DataFrame: Standardized dataframe
    """
    df_std = df.copy()
    
    # Create unique building ID
    df_std['OSEBuildingID'] = df_std['10_Digit_BBL']
    df_std['DataYear'] = 2021
    df_std['BuildingType'] = 'NonResidential'  # Assumption
    
    # Create address from street components
    df_std['Address'] = df_std['Street_Number'].astype(str) + ' ' + df_std['Street_Name'].fillna('')
    df_std['City'] = 'New York'
    df_std['State'] = 'NY'
    
    # Map Energy Star Score
    if 'Energy_Star_1-100_Score' in df_std.columns:
        df_std['ENERGYSTARScore'] = df_std['Energy_Star_1-100_Score']
    
    # Map building area
    if 'DOF_Gross_Square_Footage' in df_std.columns:
        df_std['PropertyGFABuilding(s)'] = df_std['DOF_Gross_Square_Footage']
        df_std['PropertyGFATotal'] = df_std['DOF_Gross_Square_Footage']
    
    # Map energy efficiency grade to categorical
    if 'Energy_Efficiency_Grade' in df_std.columns:
        df_std['EnergyGrade'] = df_std['Energy_Efficiency_Grade']
    
    # Add default values for missing columns
    df_std['ComplianceStatus'] = 'Unknown'
    df_std['Outlier'] = False
    
    return df_std


def combine_datasets(datasets_dict):
    """
    Combine multiple datasets into one
    
    Args:
        datasets_dict (dict): Dictionary of {dataset_name: dataframe}
        
    Returns:
        pd.DataFrame: Combined dataframe
    """
    print(f"Combining {len(datasets_dict)} datasets...")
    
    # Find common columns across all datasets
    all_columns = set()
    for df in datasets_dict.values():
        all_columns.update(df.columns)
    
    common_columns = all_columns.copy()
    for df in datasets_dict.values():
        common_columns &= set(df.columns)
    
    print(f"Found {len(common_columns)} common columns")
    
    # Create unified dataset with all columns
    unified_dfs = []
    
    for name, df in datasets_dict.items():
        df_unified = df.copy()
        
        # Add missing columns with NaN
        for col in all_columns:
            if col not in df_unified.columns:
                df_unified[col] = np.nan
        
        # Add dataset source column
        df_unified['DatasetSource'] = name
        
        # Reorder columns
        column_order = sorted(df_unified.columns)
        df_unified = df_unified[column_order]
        
        unified_dfs.append(df_unified)
        print(f"  {name}: {df_unified.shape[0]} rows")
    
    # Combine all datasets
    combined_df = pd.concat(unified_dfs, ignore_index=True, sort=False)
    
    print(f"Combined dataset shape: {combined_df.shape}")
    return combined_df


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
        'OSEBuildingID', 'BuildingName', 'PropertyName', 'Address', 'City', 'State',
        'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 'TaxParcelIdentificationNumber',
        'ComplianceStatus', 'Comments', 'Comment', 'DefaultData', 'ListOfAllPropertyUseTypes',
        '10_Digit_BBL', 'Street_Number', 'Street_Name',
        # Geographic/administrative columns that are too specific
        '2010 Census Tracts', 'Seattle Police Department Micro Community Policing Plan Areas',
        'City Council Districts', 'SPD Beats', 'Zip Codes'
    ]
    
    # Only drop columns that actually exist in the dataframe
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_columns_to_drop)
    
    print(f"Dropped {len(existing_columns_to_drop)} irrelevant columns")
    print(f"Remaining columns: {df_cleaned.shape[1]}")
    
    return df_cleaned


def handle_target_variable(df, target_priority=['SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)', 
                                               'SourceEUI(kBtu/sf)', 'SiteEUI(kBtu/sf)',
                                               'TotalGHGEmissions', 'ENERGYSTARScore']):
    """
    Extract and clean the target variable with priority order
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_priority (list): List of preferred target columns in order
        
    Returns:
        tuple: (features_df, target_series, target_name)
    """
    target_col = None
    
    # Find the first available target column from priority list
    for target in target_priority:
        if target in df.columns and df[target].notna().sum() > 100:  # At least 100 valid values
            target_col = target
            break
    
    # If no priority target found, find any numerical column
    if target_col is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].notna().sum() > 100:
                target_col = col
                break
    
    if target_col is None:
        raise ValueError("No suitable target variable found")
    
    print(f"Using '{target_col}' as target variable")
    
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Remove rows where target is null or invalid (for energy/emissions, remove non-positive)
    if 'Energy' in target_col or 'GHG' in target_col:
        valid_indices = ~(y.isnull() | (y <= 0))
    else:  # For scores, remove null only
        valid_indices = ~y.isnull()
    
    X = X[valid_indices]
    y = y[valid_indices]
    
    print(f"Removed {len(df) - len(X)} rows with invalid target values")
    print(f"Final dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, target_col


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
        unique_values = X_encoded[col].nunique()
        
        if unique_values > 200:  # Skip columns with too many unique values
            print(f"Dropping {col} - too many unique values ({unique_values})")
            X_encoded = X_encoded.drop(columns=[col])
        elif unique_values <= 1:  # Skip constant columns
            print(f"Dropping {col} - constant column")
            X_encoded = X_encoded.drop(columns=[col])
        else:
            le = LabelEncoder()
            # Handle missing values by treating them as a separate category
            X_encoded[col] = X_encoded[col].fillna('Unknown')
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            print(f"Encoded {col} - {unique_values} unique values")
    
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
    
    # Remove columns with too many missing values (>70%)
    missing_threshold = 0.7
    missing_percentages = X.isnull().sum() / len(X)
    columns_to_remove = missing_percentages[missing_percentages > missing_threshold].index
    
    if len(columns_to_remove) > 0:
        X = X.drop(columns=columns_to_remove)
        print(f"Removed {len(columns_to_remove)} columns with >{missing_threshold*100}% missing values")
    
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
    
    # Remove highly correlated features (>0.95 correlation)
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = X[numerical_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        
        if high_corr_features:
            X = X.drop(columns=high_corr_features)
            print(f"Removed {len(high_corr_features)} highly correlated features")
    
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


def preprocess_multiple_datasets(file_paths, test_size=0.2, random_state=42):
    """
    Complete data preprocessing pipeline for multiple datasets
    
    Args:
        file_paths (list): List of paths to CSV files
        test_size (float): Proportion of test set
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names, target_name)
    """
    print("="*80)
    print("STARTING MULTI-DATASET PREPROCESSING PIPELINE")
    print("="*80)
    
    # Load and standardize datasets
    datasets = {}
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
            
        df = load_data(file_path)
        if df is None:
            continue
        
        dataset_type = identify_dataset_type(df, file_path)
        print(f"Identified as: {dataset_type}")
        
        # Standardize based on dataset type
        if dataset_type == 'seattle_2015':
            df = standardize_seattle_2015(df)
        elif dataset_type == 'nyc_2021':
            df = standardize_nyc_2021(df)
        # seattle_2016 needs no standardization (base format)
        
        datasets[dataset_type] = df
        print(f"Processed {dataset_type}: {df.shape}")
    
    if not datasets:
        print("No valid datasets found!")
        return None
    
    # Combine datasets
    combined_df = combine_datasets(datasets)
    
    print(f"\nCombined dataset info:")
    print(f"Shape: {combined_df.shape}")
    print(f"Missing values: {combined_df.isnull().sum().sum()}")
    
    # Drop irrelevant columns
    combined_df = drop_irrelevant_columns(combined_df)
    
    # Handle target variable
    X, y, target_name = handle_target_variable(combined_df)
    
    # Encode categorical features
    X = encode_categorical_features(X)
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Additional cleaning
    X = feature_selection_and_cleaning(X)
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size, random_state)
    
    print("\n" + "="*80)
    print("MULTI-DATASET PREPROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Target variable: {target_name}")
    
    # Print dataset distribution
    if 'DatasetSource' in combined_df.columns:
        print(f"\nDataset distribution:")
        source_counts = combined_df['DatasetSource'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} buildings ({count/len(combined_df)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, scaler, X_train.columns.tolist(), target_name


if __name__ == "__main__":
    # Define file paths
    file_paths = [
        "data/2016-building-energy-benchmarking.csv",
        "data/2015-building-energy-benchmarking.csv", 
        "data/energy_disclosure_2021_rows.csv"
    ]
    
    print(f"Processing datasets:")
    for path in file_paths:
        print(f"  - {path}")
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
    
    # Run preprocessing
    result = preprocess_multiple_datasets(file_paths)
    
    if result is not None:
        X_train, X_test, y_train, y_test, scaler, feature_names, target_name = result
        
        # Save preprocessed data
        print("\nSaving preprocessed data...")
        X_train.to_csv("outputs/X_train.csv", index=False)
        X_test.to_csv("outputs/X_test.csv", index=False)
        y_train.to_csv("outputs/y_train.csv", index=False)
        y_test.to_csv("outputs/y_test.csv", index=False)
        
        # Save metadata
        with open("outputs/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        with open("outputs/target_info.txt", "w") as f:
            f.write(f"Target variable: {target_name}\n")
            f.write(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]\n")
        
        # Save scaler
        import joblib
        joblib.dump(scaler, "outputs/scaler.pkl")
        
        print("Multi-dataset preprocessing completed successfully!")
        print(f"Final dataset contains {X_train.shape[0] + X_test.shape[0]} buildings")
        print(f"Using {target_name} as prediction target")
    else:
        print("Multi-dataset preprocessing failed. Please check the input files.")