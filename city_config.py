"""
City-Specific Configuration for Fair Building Energy Analysis
Adjust these settings to optimize results for each city while maintaining fairness
"""

CITY_CONFIGS = {
    'seattle_2015_present': {
        'display_name': 'Seattle 2015-Present',
        'description': 'Seattle building energy benchmarking data from 2015 to present',
        
        # Target variable preferences (in order of priority)
        'target_patterns': [
            'SiteEUI(kBtu/sf)', 'SiteEUI', 'Site EUI (kBtu/sf)', 'Site EUI',
            'SourceEUI(kBtu/sf)', 'SourceEUI', 'Source EUI (kBtu/sf)', 'Source EUI',
            'ENERGYSTARScore', 'ENERGY STAR Score', 'Energy Star Score',
            'TotalGHGEmissions', 'Total GHG Emissions', 'GHG Emissions',
            'Electricity(kWh)', 'Electricity(kBtu)', 'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)'
        ],
        
        # Data quality thresholds
        'min_threshold': 50,  # Minimum valid samples required
        'outlier_percentiles': (5, 95),  # Remove extreme 5% from each end
        'max_missing_percent': 70,  # Drop columns with >70% missing values
        'min_variance_threshold': 0.001,  # Minimum variance for numeric columns
        
        # Column management
        'irrelevant_cols': [
            'OSEBuildingID', 'BuildingName', 'PropertyName', 'Address', 'City', 'State',
            'ZipCode', 'CouncilDistrictCode', 'Neighborhood', 'ComplianceStatus', 'Comments',
            'TaxParcelIdentificationNumber', 'Location', 'Demolished'
        ],
        'building_type_col': 'LargestPropertyUseType',
        'year_col': 'DataYear',
        
        # Model training preferences
        'classification_min_samples': 100,
        'use_hyperparameter_tuning': True,
        'cross_validation_folds': 5
    },
    
    'chicago_energy': {
        'display_name': 'Chicago Energy Benchmarking',
        'description': 'Chicago commercial building energy benchmarking data',
        
        # Target variable preferences - optimized for Chicago data structure
        'target_patterns': [
            'Electricity Use (kBtu)', 'Electricity Use', 'Natural Gas Use (kBtu)', 'Natural Gas Use',
            'Site EUI (kBtu/sq ft)', 'Site EUI', 'Site Energy Use Intensity',
            'ENERGY STAR Score', 'Energy Star Rating', 'Energy Star Score', 'Chicago Energy Rating',
            'Total GHG Emissions (Metric Tons CO2e)', 'Total GHG Emissions', 'GHG Emissions',
            'Weather Normalized Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)'
        ],
        
        # More lenient thresholds for Chicago
        'min_threshold': 15,  # Lower threshold due to smaller dataset
        'outlier_percentiles': (2, 98),  # Less aggressive outlier removal
        'max_missing_percent': 80,  # Allow more missing values
        'min_variance_threshold': 0.0001,  # Lower variance threshold
        
        # Chicago-specific columns to remove
        'irrelevant_cols': [
            'ID', 'Property Name', 'Reporting Status', 'Address', 'City', 'State', 
            'ZIP Code', 'Community Area', 'Latitude', 'Longitude', 'Location', 'Row_ID',
            'Exempt From Chicago Energy Rating'
        ],
        'building_type_col': 'Primary Property Type',
        'year_col': 'Data Year',
        
        # Adjusted for smaller dataset
        'classification_min_samples': 50,  # Lower threshold
        'use_hyperparameter_tuning': False,  # Skip for speed
        'cross_validation_folds': 3  # Fewer folds
    },
    
    'washington_dc': {
        'display_name': 'Washington DC Building Energy',
        'description': 'Washington DC building energy disclosure data',
        
        # Target variable preferences - DC specific
        'target_patterns': [
            'Site EUI (kBtu/sq ft)', 'Site EUI', 'Site Energy Use Intensity',
            'ENERGY STAR Score', 'Energy Star Rating', 'Portfolio Manager Score',
            'Total CO2 Emissions (Metric Tons)', 'Total CO2 Emissions', 'GHG Emissions', 'Total Emissions',
            'Electricity Use (kBtu)', 'Natural Gas Use (kBtu)', 'Energy Use',
            'Weather Normalized Site EUI (kBtu/sq ft)', 'Source EUI (kBtu/sq ft)'
        ],
        
        # Most lenient thresholds for DC (smallest dataset)
        'min_threshold': 10,  # Very low threshold
        'outlier_percentiles': (1, 99),  # Minimal outlier removal
        'max_missing_percent': 85,  # Very tolerant of missing data
        'min_variance_threshold': 0.00001,  # Very low variance threshold
        
        # DC-specific columns to remove
        'irrelevant_cols': [
            'Portfolio Manager ID', 'Property Name', 'Address', 'City', 'State',
            'Postal Code', 'Ward', 'Agency', 'Ownership', 'Reporting Status'
        ],
        'building_type_col': 'Property Type',
        'year_col': 'Year',
        
        # Very lenient for DC
        'classification_min_samples': 30,  # Very low threshold
        'use_hyperparameter_tuning': False,  # Skip for speed
        'cross_validation_folds': 3  # Fewer folds
    }
}

# Universal building type mappings for standardization
UNIVERSAL_BUILDING_TYPES = {
    # Office variants
    'office': 'Office',
    'commercial': 'Office',
    'business': 'Office',
    'administrative': 'Office',
    
    # Residential variants
    'residential': 'Residential',
    'multifamily': 'Residential', 
    'apartment': 'Residential',
    'housing': 'Residential',
    'condominium': 'Residential',
    'senior': 'Residential',
    
    # Retail variants
    'retail': 'Retail',
    'store': 'Retail',
    'shopping': 'Retail',
    'mall': 'Retail',
    'supermarket': 'Retail',
    'grocery': 'Retail',
    
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
    'elementary': 'Education',
    'high school': 'Education',
    'library': 'Education',
    
    # Industrial variants
    'warehouse': 'Industrial',
    'manufacturing': 'Industrial',
    'industrial': 'Industrial',
    'distribution': 'Industrial',
    'storage': 'Industrial',
    'logistics': 'Industrial',
    
    # Hospitality variants
    'hotel': 'Hospitality',
    'lodging': 'Hospitality',
    'motel': 'Hospitality',
    'inn': 'Hospitality',
    
    # Government variants
    'government': 'Government',
    'federal': 'Government',
    'municipal': 'Government',
    'public': 'Government',
    'courthouse': 'Government',
    'city hall': 'Government',
    
    # Entertainment/Recreation
    'entertainment': 'Entertainment',
    'recreation': 'Entertainment',
    'theater': 'Entertainment',
    'stadium': 'Entertainment',
    'gym': 'Entertainment',
    'fitness': 'Entertainment',
    
    # Religious
    'religious': 'Religious',
    'church': 'Religious',
    'worship': 'Religious',
    'temple': 'Religious',
    'mosque': 'Religious'
}

# Performance expectations by city (for comparison)
PERFORMANCE_BENCHMARKS = {
    'seattle_2015_present': {
        'min_acceptable_r2': 0.60,
        'good_r2': 0.75,
        'excellent_r2': 0.85,
        'min_samples_expected': 500,
        'max_processing_time': 60  # seconds
    },
    'chicago_energy': {
        'min_acceptable_r2': 0.45,  # Lower expectations due to data challenges
        'good_r2': 0.65,
        'excellent_r2': 0.80,
        'min_samples_expected': 100,
        'max_processing_time': 45
    },
    'washington_dc': {
        'min_acceptable_r2': 0.40,  # Lowest expectations - smallest dataset
        'good_r2': 0.60,
        'excellent_r2': 0.75,
        'min_samples_expected': 50,
        'max_processing_time': 30
    }
}

# Feature engineering preferences
FEATURE_ENGINEERING = {
    'create_building_age': True,  # Calculate age from year built
    'create_efficiency_ratios': True,  # EUI per square foot ratios
    'create_size_categories': True,  # Small/Medium/Large building categories
    'create_vintage_categories': True,  # Building age categories
    'normalize_by_climate': False,  # Skip climate normalization for now
    'create_density_features': False  # Skip density features for now
}

def get_city_config(dataset_type):
    """
    Get configuration for a specific city dataset
    
    Args:
        dataset_type (str): Type of dataset ('seattle_2015_present', 'chicago_energy', 'washington_dc')
        
    Returns:
        dict: Configuration dictionary for the city
    """
    return CITY_CONFIGS.get(dataset_type, CITY_CONFIGS['seattle_2015_present'])

def get_building_type_mappings():
    """
    Get universal building type mappings
    
    Returns:
        dict: Building type mapping dictionary
    """
    return UNIVERSAL_BUILDING_TYPES

def get_performance_benchmark(dataset_type):
    """
    Get performance benchmarks for a specific city
    
    Args:
        dataset_type (str): Type of dataset
        
    Returns:
        dict: Performance benchmark dictionary
    """
    return PERFORMANCE_BENCHMARKS.get(dataset_type, PERFORMANCE_BENCHMARKS['seattle_2015_present'])

def print_city_config_summary():
    """Print a summary of all city configurations"""
    
    print("="*80)
    print("CITY CONFIGURATION SUMMARY")
    print("="*80)
    
    for city_key, config in CITY_CONFIGS.items():
        print(f"\n🏙️ {config['display_name'].upper()}")
        print(f"   Dataset key: {city_key}")
        print(f"   Description: {config['description']}")
        print(f"   Min threshold: {config['min_threshold']} samples")
        print(f"   Outlier removal: {config['outlier_percentiles'][0]}%-{config['outlier_percentiles'][1]}%")
        print(f"   Max missing: {config['max_missing_percent']}%")
        print(f"   Classification min: {config['classification_min_samples']} samples")
        print(f"   Hyperparameter tuning: {'Yes' if config['use_hyperparameter_tuning'] else 'No'}")
        
        # Performance expectations
        benchmarks = get_performance_benchmark(city_key)
        print(f"   Expected R² range: {benchmarks['min_acceptable_r2']:.2f} - {benchmarks['excellent_r2']:.2f}")
        print(f"   Target samples: ≥{benchmarks['min_samples_expected']}")
    
    print(f"\n🏢 BUILDING TYPE STANDARDIZATION:")
    print(f"   Total mappings: {len(UNIVERSAL_BUILDING_TYPES)}")
    print(f"   Standard types: Office, Residential, Retail, Healthcare, Education,")
    print(f"                   Industrial, Hospitality, Government, Entertainment, Religious")
    
    print(f"\n⚙️ FEATURE ENGINEERING:")
    for feature, enabled in FEATURE_ENGINEERING.items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")

def validate_city_configs():
    """Validate that all city configurations are properly set up"""
    
    print("🔍 Validating city configurations...")
    
    required_keys = [
        'display_name', 'description', 'target_patterns', 'min_threshold',
        'outlier_percentiles', 'irrelevant_cols', 'building_type_col', 'year_col'
    ]
    
    issues = []
    
    for city_key, config in CITY_CONFIGS.items():
        for key in required_keys:
            if key not in config:
                issues.append(f"❌ {city_key}: Missing '{key}'")
        
        # Validate thresholds
        if config.get('min_threshold', 0) <= 0:
            issues.append(f"❌ {city_key}: min_threshold must be > 0")
        
        # Validate outlier percentiles
        outlier_range = config.get('outlier_percentiles', (0, 100))
        if outlier_range[0] >= outlier_range[1]:
            issues.append(f"❌ {city_key}: Invalid outlier_percentiles range")
    
    if issues:
        print("⚠️ Configuration issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ All city configurations are valid")
        return True

# Custom configuration adjustments for specific issues
CUSTOM_ADJUSTMENTS = {
    'chicago_energy': {
        'note': 'Chicago dataset may have industrial buildings with extreme energy usage',
        'custom_outlier_strategy': 'building_type_aware',  # Different outlier removal per building type
        'prefer_ratio_targets': True,  # Prefer EUI over absolute consumption
        'extra_data_cleaning': True
    },
    'washington_dc': {
        'note': 'DC dataset is smallest, government buildings may have different patterns',
        'custom_outlier_strategy': 'conservative',  # Very conservative outlier removal
        'prefer_score_targets': True,  # Prefer Energy Star scores over consumption
        'skip_correlation_removal': True  # Don't remove correlated features
    },
    'seattle_2015_present': {
        'note': 'Seattle dataset is most complete, use as baseline',
        'custom_outlier_strategy': 'standard',
        'prefer_consumption_targets': True,  # Prefer direct consumption metrics
        'full_feature_engineering': True
    }
}

def get_custom_adjustments(dataset_type):
    """Get custom adjustments for specific dataset challenges"""
    return CUSTOM_ADJUSTMENTS.get(dataset_type, {})

if __name__ == "__main__":
    print("🏙️ Building Energy Analysis - City Configuration")
    print_city_config_summary()
    print("\n" + "="*80)
    validate_city_configs()
    
    print(f"\n💡 USAGE:")
    print(f"   from city_config import get_city_config")
    print(f"   config = get_city_config('chicago_energy')")
    print(f"   min_samples = config['min_threshold']")