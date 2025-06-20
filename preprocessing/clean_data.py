"""
Enhanced Data Preprocessing Script for Multiple Building Energy Datasets
COMPLETE FIXED VERSION - Fair preprocessing for Seattle, Chicago, and Washington DC datasets
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
    Enhanced dataset identification with better column-based detection
    
    Args:
        df (pd.DataFrame): Input dataframe
        file_path (str): Path to identify dataset
        
    Returns:
        str: Dataset type
    """
    filename = os.path.basename(file_path).lower()
    
    # Primary identification by filename
    if 'seattle' in filename:
        return 'seattle_2015_present'
    elif 'chicago' in filename:
        return 'chicago_energy'
    elif 'washington' in filename or 'dc' in filename:
        return 'washington_dc'
    
    # Enhanced column-based identification
    columns_lower = [col.lower() for col in df.columns]
    column_text = ' '.join(columns_lower)
    
    # Washington DC signatures
    dc_indicators = ['ward', 'portfolio manager', 'federal', 'district']
    if any(indicator in column_text for indicator in dc_indicators):
        return 'washington_dc'
    
    # Chicago signatures  
    chicago_indicators = ['chicago energy benchmarking', 'community area', 'chicago']
    if any(indicator in column_text for indicator in chicago_indicators):
        return 'chicago_energy'
    
    # Seattle signatures
    seattle_indicators = ['ose building', 'compliance status', 'seattle']
    if any(indicator in column_text for indicator in seattle_indicators):
        return 'seattle_2015_present'
    
    print(f"Warning: Could not identify dataset type for {filename}, defaulting to Seattle")
    return 'seattle_2015_present'


def get_fair_city_config(dataset_type):
    """
    Get FAIR city-specific configuration - BALANCED approach
    """
    configs = {
        'seattle_2015_present': {
            'target_patterns': [
                'SiteEUI(kBtu/sf)', 'SiteEUI', 'Site EUI (kBtu/sf)', 'Site EUI',
                'SourceEUI(kBtu/sf)', 'SourceEUI', 'Source EUI (kBtu/sf)', 'Source EUI',
                'ENERGYSTARScore', 'ENERGY STAR Score', 'Energy Star Score',
                'TotalGHGEmissions', 'Total GHG Emissions', 'GHG Emissions',
                'Electricity(kWh)', 'Electricity(kBtu)', 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)'
            ],
            'min_threshold': 50,  # REDUCED from 100 for fairness
            'outlier_percentiles': (5, 95),  # LESS aggressive outlier removal
            'irrelevant_cols': [
                'OSEBuildingID', 'BuildingName', 'PropertyName', 'Address', 'City', 'State',
                'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 'ComplianceStatus', 'Comments',
                'TaxParcelIdentificationNumber', 'Location', 'Demolished'
            ],
            'building_type_col': 'LargestPropertyUseType',
            'year_col': 'DataYear'
        },
        
        'chicago_energy': {
            'target_patterns': [
                'Electricity Use (kBtu)', 'Electricity Use', 'Natural Gas Use (kBtu)', 'Natural Gas Use',
                'Site EUI (kBtu/sq ft)', 'Site EUI', 'Site Energy Use Intensity',
                'ENERGY STAR Score', 'Energy Star Rating', 'Energy Star Score', 'Chicago Energy Rating',
                'Total GHG Emissions (Metric Tons CO2e)', 'Total GHG Emissions', 'GHG Emissions',
                'Weather Normalized Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)'
            ],
            'min_threshold': 15,  # MUCH lower threshold for Chicago
            'outlier_percentiles': (2, 98),  # LESS aggressive for Chicago
            'irrelevant_cols': [
                'ID', 'Property Name', 'Reporting Status', 'Address', 'City', 'State', 
                'ZIP Code', 'Community Area', 'Latitude', 'Longitude', 'Location', 'Row_ID',
                'Exempt From Chicago Energy Rating'
            ],
            'building_type_col': 'Primary Property Type',
            'year_col': 'Data Year'
        },
        
        'washington_dc': {
            'target_patterns': [
                'Site EUI (kBtu/sq ft)', 'Site EUI', 'Site Energy Use Intensity',
                'ENERGY STAR Score', 'Energy Star Rating', 'Portfolio Manager Score',
                'Total CO2 Emissions (Metric Tons)', 'Total CO2 Emissions', 'GHG Emissions', 'Total Emissions',
                'Electricity Use (kBtu)', 'Natural Gas Use (kBtu)', 'Energy Use',
                'Weather Normalized Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)'
            ],
            'min_threshold': 10,  # VERY low threshold for DC (smallest dataset)
            'outlier_percentiles': (1, 99),  # LEAST aggressive for DC
            'irrelevant_cols': [
                'Portfolio Manager ID', 'Property Name', 'Address', 'City', 'State',
                'Postal Code', 'Ward', 'Agency', 'Ownership', 'Reporting Status'
            ],
            'building_type_col': 'Property Type',
            'year_col': 'Year'
        }
    }
    
    return configs.get(dataset_type, configs['seattle_2015_present'])


def find_target_variable_enhanced(df, dataset_type=None):
    """
    FAIR target variable detection - equal treatment for all cities
    
    Args:
        df (pd.DataFrame): Input dataframe
        dataset_type (str): Type of dataset
        
    Returns:
        str or None: Name of target variable column
    """
    print(f"Enhanced target detection for {dataset_type}")
    print(f"Available columns: {len(df.columns)}")
    
    config = get_fair_city_config(dataset_type)
    energy_patterns = config['target_patterns']
    min_valid_threshold = config['min_threshold']
    
    print(f"Using minimum threshold: {min_valid_threshold} for {dataset_type}")
    
    # Try each pattern in order - FAIR approach
    candidates = []
    
    for pattern in energy_patterns:
        # Exact match first
        if pattern in df.columns:
            valid_count = df[pattern].notna().sum()
            numeric_count = pd.to_numeric(df[pattern], errors='coerce').notna().sum()
            
            if valid_count >= min_valid_threshold and numeric_count >= min_valid_threshold:
                variance = pd.to_numeric(df[pattern], errors='coerce').var()
                if pd.notna(variance) and variance > 0:
                    candidates.append((pattern, valid_count, variance))
                    print(f"✓ Candidate: {pattern} ({valid_count} valid, variance: {variance:.2f})")
    
    # Case-insensitive matching if no exact match
    if not candidates:
        df_columns_lower = [col.lower() for col in df.columns]
        for pattern in energy_patterns:
            pattern_lower = pattern.lower()
            for i, col_lower in enumerate(df_columns_lower):
                if pattern_lower == col_lower or pattern_lower in col_lower:
                    actual_col = df.columns[i]
                    valid_count = df[actual_col].notna().sum()
                    numeric_count = pd.to_numeric(df[actual_col], errors='coerce').notna().sum()
                    
                    if valid_count >= min_valid_threshold and numeric_count >= min_valid_threshold:
                        variance = pd.to_numeric(df[actual_col], errors='coerce').var()
                        if pd.notna(variance) and variance > 0:
                            candidates.append((actual_col, valid_count, variance))
                            print(f"✓ Case-insensitive candidate: {actual_col}")
    
    # Fuzzy matching for any numerical column with energy keywords
    if not candidates:
        energy_keywords = ['eui', 'energy', 'star', 'ghg', 'emission', 'co2', 'electricity', 'gas', 'kbtu', 'kwh']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in energy_keywords):
                if df[col].dtype in ['int64', 'float64'] or pd.to_numeric(df[col], errors='coerce').notna().sum() > min_valid_threshold:
                    valid_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if valid_count >= min_valid_threshold:
                        try:
                            variance = pd.to_numeric(df[col], errors='coerce').var()
                            if pd.notna(variance) and variance > 0:
                                candidates.append((col, valid_count, variance))
                                print(f"✓ Fuzzy candidate: {col}")
                        except:
                            continue
    
    # Final fallback - any numeric column with sufficient data
    if not candidates:
        print("Using fallback: searching all numeric columns...")
        for col in df.columns:
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                valid_count = numeric_series.notna().sum()
                if valid_count >= min_valid_threshold:
                    variance = numeric_series.var()
                    if pd.notna(variance) and variance > 0:
                        candidates.append((col, valid_count, variance))
                        print(f"✓ Fallback candidate: {col}")
            except:
                continue
    
    if candidates:
        # Select best candidate (highest valid count, then highest variance)
        best_candidate = max(candidates, key=lambda x: (x[1], x[2]))
        target_col = best_candidate[0]
        print(f"✅ Selected target: {target_col} ({best_candidate[1]} valid values)")
        return target_col
    
    print(f"❌ No suitable target found for {dataset_type}")
    return None


def apply_fair_outlier_removal(df, dataset_type, target_col):
    """
    Apply FAIR outlier removal - adaptive based on dataset characteristics
    """
    config = get_fair_city_config(dataset_type)
    low_pct, high_pct = config['outlier_percentiles']
    
    print(f"{dataset_type}: Applying FAIR outlier removal ({low_pct}%-{high_pct}% range)")
    
    original_shape = df.shape[0]
    
    # Focus outlier removal mainly on target variable
    if target_col in df.columns:
        target_data = pd.to_numeric(df[target_col], errors='coerce')
        q_low = target_data.quantile(low_pct / 100)
        q_high = target_data.quantile(high_pct / 100)
        
        # Remove extreme outliers in target
        target_mask = (target_data >= q_low) & (target_data <= q_high)
        df = df[target_mask]
        
        print(f"Target outlier removal: kept {df.shape[0]}/{original_shape} rows")
    
    # Light outlier removal for other numeric columns (only extreme cases)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_col and df[col].notna().sum() > 20:
            # Very conservative outlier removal for non-target columns
            q1 = df[col].quantile(0.001)  # Only remove extreme 0.1% outliers
            q99 = df[col].quantile(0.999)
            df = df[(df[col] >= q1) & (df[col] <= q99)]
    
    final_shape = df.shape[0]
    removed_count = original_shape - final_shape
    removal_pct = (removed_count / original_shape) * 100
    
    print(f"Total outlier removal: {removed_count} rows ({removal_pct:.1f}%)")
    
    # Ensure we don't remove too much data
    if removal_pct > 80:
        print(f"⚠️ Warning: Removed {removal_pct:.1f}% of data - this may be too aggressive")
    
    return df


def standardize_building_types(building_type_series, dataset_type):
    """
    Standardize building types across different cities for fair comparison
    
    Args:
        building_type_series (pd.Series): Building type column
        dataset_type (str): Type of dataset
        
    Returns:
        pd.Series: Standardized building types
    """
    # Universal building type mapping
    universal_mapping = {
        # Office variants
        'office': 'Office',
        'commercial': 'Office',
        'business': 'Office',
        
        # Residential variants
        'residential': 'Residential',
        'multifamily': 'Residential', 
        'apartment': 'Residential',
        'housing': 'Residential',
        'condominium': 'Residential',
        
        # Retail variants
        'retail': 'Retail',
        'store': 'Retail',
        'shopping': 'Retail',
        'mall': 'Retail',
        
        # Healthcare variants
        'hospital': 'Healthcare',
        'medical': 'Healthcare',
        'health': 'Healthcare',
        'clinic': 'Healthcare',
        
        # Education variants
        'school': 'Education',
        'education': 'Education',
        'university': 'Education',
        'college': 'Education',
        'k-12': 'Education',
        
        # Industrial variants
        'warehouse': 'Industrial',
        'manufacturing': 'Industrial',
        'industrial': 'Industrial',
        'distribution': 'Industrial',
        'storage': 'Industrial',
        
        # Hospitality variants
        'hotel': 'Hospitality',
        'lodging': 'Hospitality',
        'motel': 'Hospitality',
        
        # Government variants
        'government': 'Government',
        'federal': 'Government',
        'municipal': 'Government',
        'public': 'Government'
    }
    
    standardized = building_type_series.copy()
    
    for idx, building_type in enumerate(building_type_series):
        if pd.isna(building_type):
            standardized.iloc[idx] = 'Other'
            continue
            
        building_type_lower = str(building_type).lower().strip()
        
        # Try exact mapping first
        mapped = False
        for key, value in universal_mapping.items():
            if key in building_type_lower:
                standardized.iloc[idx] = value
                mapped = True
                break
        
        if not mapped:
            standardized.iloc[idx] = 'Other'
    
    return standardized


def preprocess_data_city_specific(df, dataset_type):
    """
    City-specific data preprocessing for fair comparison
    
    Args:
        df (pd.DataFrame): Input dataframe
        dataset_type (str): Type of dataset
        
    Returns:
        tuple: (df, target_col)
    """
    print(f"Applying city-specific preprocessing for {dataset_type}")
    config = get_fair_city_config(dataset_type)
    
    # Find and validate target variable
    target_col = find_target_variable_enhanced(df, dataset_type)
    
    if target_col is None:
        print(f"ERROR: No suitable target variable found for {dataset_type}")
        print("Available columns:")
        for i, col in enumerate(df.columns):
            non_null_count = df[col].notna().sum()
            dtype = df[col].dtype
            print(f"  {i+1:2d}. {col} ({dtype}) - {non_null_count} non-null values")
        raise ValueError(f"No suitable target variable found for {dataset_type}")
    
    print(f"✅ Using target variable: {target_col}")
    
    # Clean target variable FIRST (before other preprocessing)
    print(f"Cleaning target variable: {target_col}")
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Remove rows with invalid target values
    if any(keyword in target_col.lower() for keyword in ['eui', 'energy', 'consumption', 'use']):
        # For energy metrics, remove negative and zero values
        valid_target_mask = (df[target_col] > 0) & df[target_col].notna()
    elif any(keyword in target_col.lower() for keyword in ['star', 'score', 'rating']):
        # For scores, keep positive values
        valid_target_mask = (df[target_col] >= 1) & df[target_col].notna()
    else:
        # For other metrics, just remove null values
        valid_target_mask = df[target_col].notna()
    
    original_count = len(df)
    df = df[valid_target_mask]
    removed_invalid = original_count - len(df)
    print(f"Removed {removed_invalid} rows with invalid target values")
    
    # Apply FAIR outlier removal
    df = apply_fair_outlier_removal(df, dataset_type, target_col)
    
    # Standardize building types if column exists
    building_type_col = config['building_type_col']
    if building_type_col in df.columns:
        df[building_type_col] = standardize_building_types(df[building_type_col], dataset_type)
        print(f"Standardized building types in: {building_type_col}")
    
    # Handle year column if exists
    year_col = config['year_col']
    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        print(f"Converted {year_col} to numeric")
    
    return df, target_col


def preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Enhanced preprocessing with city-specific adaptations for fair comparison
    
    Args:
        file_path (str): Path to the CSV file
        test_size (float): Test set proportion
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print(f"Loading and preprocessing: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    
    # Identify dataset type
    dataset_type = identify_dataset_type(df, file_path)
    print(f"Identified as: {dataset_type}")
    
    # Apply city-specific preprocessing
    df, target_col = preprocess_data_city_specific(df, dataset_type)
    
    if len(df) < 10:
        raise ValueError(f"Too few samples remaining after preprocessing: {len(df)}")
    
    # Separate features and target
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    print(f"After preprocessing: {X.shape}")
    print(f"Target variable: {target_col}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Get config for irrelevant columns
    config = get_fair_city_config(dataset_type)
    irrelevant_cols = config['irrelevant_cols']
    
    # Drop irrelevant columns
    existing_irrelevant = [col for col in irrelevant_cols if col in X.columns]
    if existing_irrelevant:
        X = X.drop(columns=existing_irrelevant)
        print(f"Dropped {len(existing_irrelevant)} irrelevant columns")
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        unique_count = X[col].nunique()
        if unique_count > 100:  # Too many categories
            X = X.drop(columns=[col])
            print(f"Dropped {col} - too many categories ({unique_count})")
        elif unique_count <= 1:  # Constant column
            X = X.drop(columns=[col])
            print(f"Dropped {col} - constant column")
        else:
            # Label encoding
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            try:
                X[col] = le.fit_transform(X[col].astype(str))
                print(f"Encoded categorical: {col} ({unique_count} categories)")
            except Exception as e:
                print(f"Warning: Could not encode {col}, dropping: {e}")
                X = X.drop(columns=[col])
    
    # Handle missing values in numerical columns
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
    
    # Remove highly correlated features (>0.95 correlation)
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
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    feature_names = X_train.columns.tolist()
    
    print(f"FAIR preprocessing results for {dataset_type}:")
    print(f"✅ Train set: {X_train_scaled.shape}")
    print(f"✅ Test set: {X_test_scaled.shape}")
    print(f"✅ Target range: [{y.min():.0f}, {y.max():.0f}]")
    print(f"✅ Features: {len(feature_names)} total")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ===================================
# LEGACY FUNCTIONS (for compatibility)
# ===================================

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


# ===================================
# LEGACY HELPER FUNCTIONS
# ===================================

def create_energy_efficiency_labels(energy_series):
    """
    Create energy efficiency labels from continuous energy values
    
    Args:
        energy_series (pd.Series): Energy consumption values
        
    Returns:
        pd.Series: Categorical labels (Excellent, Good, Average, Poor)
    """
    # Use quartiles to create balanced categories
    q1 = energy_series.quantile(0.25)
    q2 = energy_series.quantile(0.50)
    q3 = energy_series.quantile(0.75)
    
    def assign_label(value):
        if pd.isna(value):
            return 'Unknown'
        elif value <= q1:
            return 'Excellent'  # Lowest 25% - most efficient
        elif value <= q2:
            return 'Good'       # 25-50%
        elif value <= q3:
            return 'Average'    # 50-75%
        else:
            return 'Poor'       # Top 25% - least efficient
    
    return energy_series.apply(assign_label)


def get_city_config(dataset_type):
    """Backward compatibility function"""
    return get_fair_city_config(dataset_type)


def find_target_variable(df, dataset_type=None):
    """Backward compatibility function"""
    return find_target_variable_enhanced(df, dataset_type)


print("✅ Enhanced Clean Data Module Loaded Successfully!")
print("🏙️ Features: Fair preprocessing for Seattle, Chicago, and Washington DC")
print("⚖️ Adaptive thresholds and outlier removal for each city")
print("🎯 Smart target variable detection with multiple fallbacks")