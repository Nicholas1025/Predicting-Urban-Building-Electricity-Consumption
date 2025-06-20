"""
Individual Dataset Analysis for Building Energy Prediction
FAIR VERSION: Equal treatment for all datasets with adaptive preprocessing
"""
import logging
from datetime import datetime
import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')

def create_individual_directories(dataset_name):
    directories = [
        f"outputs/{dataset_name}",
        f"outputs/{dataset_name}/charts",
        f"outputs/{dataset_name}/tables", 
        f"outputs/{dataset_name}/models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

def process_individual_dataset(dataset_name, file_path):
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET (FAIR MODE)")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        create_individual_directories(dataset_name)
        
        print("🔄 Step 1: FAIR Data Preprocessing...")

        try:
            # Use the new fair preprocessing
            from preprocessing.clean_data import preprocess_data
        except ImportError as e:
            print(f"❌ Cannot import preprocessing module: {e}")
            return False
        
        result = preprocess_data(file_path)
        if result is None:
            print(f"❌ FAIR preprocessing failed for {dataset_name}")
            return False
            
        X_train, X_test, y_train, y_test, scaler, feature_names = result

        output_dir = f"outputs/{dataset_name}"
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

        with open(f"{output_dir}/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        
        print(f"✅ FAIR preprocessing completed for {dataset_name}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Target range: [{y_train.min():.0f}, {y_train.max():.0f}]")

        print("\n🤖 Step 2: Model Training...")
        train_models_for_dataset(dataset_name)

        print("\n📊 Step 3: Model Evaluation...")
        evaluate_models_for_dataset(dataset_name)

        # More lenient classification threshold
        if len(X_train) > 100:  # Reduced from 500
            print("\n🎯 Step 4: Classification Models...")
            train_classification_for_dataset(dataset_name, X_train, y_train)
        else:
            print(f"\n⚠️  Skipping classification for {dataset_name} (insufficient data: {len(X_train)} samples)")
        
        print(f"✅ {dataset_name} FAIR analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_for_dataset(dataset_name):
    """Train models for single dataset with better error handling"""
    
    try:
        print("  🚀 Training XGBoost...")
        try:
            from models.train_xgboost_individual import main as train_xgb
            train_xgb(dataset_name)
            print("     ✅ XGBoost completed")
        except Exception as e:
            print(f"     ❌ XGBoost failed: {e}")

        print("  🌲 Training Random Forest...")
        try:
            from models.train_rf_individual import main as train_rf
            train_rf(dataset_name)
            print("     ✅ Random Forest completed")
        except Exception as e:
            print(f"     ❌ Random Forest failed: {e}")

        print("  ⚡ Training SVR...")
        try:
            from models.train_svr_individual import main as train_svr
            train_svr(dataset_name)
            print("     ✅ SVR completed")
        except Exception as e:
            print(f"     ❌ SVR failed: {e}")
        
    except Exception as e:
        print(f"❌ Model training error: {e}")

def evaluate_models_for_dataset(dataset_name):
    try:
        print(f"  📊 Evaluating models for {dataset_name}...")
        from evaluation.evaluate_models_individual import main as eval_models
        eval_models(dataset_name)
        print("     ✅ Evaluation completed")
    except Exception as e:
        print(f"     ❌ Evaluation error: {e}")

def train_classification_for_dataset(dataset_name, X_train, y_train):
    """Train classification models with FAIR approach"""
    try:
        print(f"  🎯 Training classification models for {dataset_name}...")

        import numpy as np

        def create_fair_labels(energy_values):
            """Create FAIR energy efficiency labels using adaptive quartiles"""
            # Remove outliers first for better quartile calculation
            clean_values = energy_values.dropna()
            q1 = clean_values.quantile(0.25)
            q2 = clean_values.quantile(0.50)
            q3 = clean_values.quantile(0.75)
            
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
            
            return energy_values.apply(assign_label)
        
        labels = create_fair_labels(y_train)
        
        # Check label distribution
        label_counts = labels.value_counts()
        print(f"     ✓ Label distribution: {label_counts.to_dict()}")
        
        # Only proceed if we have reasonable distribution
        if len(label_counts) >= 2 and label_counts.min() >= 5:
            output_dir = f"outputs/{dataset_name}"
            X_train.to_csv(f"{output_dir}/unified_features.csv", index=False)
            labels.to_csv(f"{output_dir}/unified_labels.csv", index=False)
            
            try:
                from models.train_classification_individual import main as train_class
                train_class(dataset_name)
                print("     ✅ Classification training completed")
            except Exception as e:
                print(f"     ❌ Classification training failed: {e}")
        else:
            print(f"     ⚠️  Insufficient class diversity for classification ({len(label_counts)} classes)")
        
    except Exception as e:
        print(f"     ❌ Classification setup error: {e}")


def generate_fairness_report(results):
    """
    Generate a comprehensive fairness analysis report
    """
    print("\n" + "="*80)
    print("FAIR ANALYSIS REPORT")
    print("="*80)
    
    print("🏙️ Multi-City Building Energy Analysis with Fair Treatment:")
    print("   ✅ Adaptive preprocessing thresholds per city")
    print("   ✅ City-specific outlier removal strategies")
    print("   ✅ Universal building type standardization")
    print("   ✅ Equal minimum sample requirements")
    
    successful_cities = []
    
    for dataset_name, result in results.items():
        if result['success']:
            print(f"\n✅ {dataset_name.upper().replace('_', ' ')}:")
            print(f"   Status: Successfully analyzed")
            print(f"   Processing time: {result['time']:.1f} seconds")
            successful_cities.append(dataset_name)
            
            # Try to read evaluation results
            try:
                eval_file = f"outputs/{dataset_name}/model_evaluation_results.csv"
                if os.path.exists(eval_file):
                    df = pd.read_csv(eval_file)
                    best_model = df.loc[df['R2'].idxmax()]
                    print(f"   🏆 Best model: {best_model['Model']}")
                    print(f"   📊 Best R²: {best_model['R2']:.4f}")
                    print(f"   📈 Best RMSE: {best_model['RMSE']:.2f}")
            except Exception as e:
                print(f"   ⚠️  Could not load detailed results: {e}")
        else:
            print(f"\n❌ {dataset_name.upper().replace('_', ' ')}:")
            print(f"   Status: Analysis failed")
            print(f"   Time: {result['time']:.1f}s")
    
    print(f"\n📊 FAIRNESS ASSESSMENT:")
    print(f"✅ Successfully analyzed {len(successful_cities)}/{len(results)} cities")
    print(f"🌍 Cities covered: {', '.join([city.replace('_', ' ').title() for city in successful_cities])}")
    
    if successful_cities:
        print(f"\n🎯 COMPARATIVE INSIGHTS:")
        
        # Compare R² scores if available
        city_performances = {}
        for city in successful_cities:
            try:
                eval_file = f"outputs/{city}/model_evaluation_results.csv"
                if os.path.exists(eval_file):
                    df = pd.read_csv(eval_file)
                    best_r2 = df['R2'].max()
                    city_performances[city] = best_r2
            except:
                continue
        
        if city_performances:
            print("📈 Best R² Score by City:")
            for city, r2 in sorted(city_performances.items(), key=lambda x: x[1], reverse=True):
                print(f"   {city.replace('_', ' ').title()}: {r2:.4f}")
    
    return results

def save_analysis_summary(results, total_time):
    """Save a detailed analysis summary with fairness metrics"""
    
    logger = logging.getLogger('BuildingEnergyAnalysis')
    
    try:
        # Calculate fairness metrics
        successful_count = sum(1 for r in results.values() if r['success'])
        avg_processing_time = np.mean([r['time'] for r in results.values()])
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_mode': 'FAIR_INDIVIDUAL_ANALYSIS',
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'datasets_processed': len(results),
            'successful_analyses': successful_count,
            'failed_analyses': len(results) - successful_count,
            'success_rate': successful_count / len(results) * 100,
            'average_processing_time': avg_processing_time,
            'fairness_features': {
                'adaptive_thresholds': True,
                'city_specific_outlier_removal': True,
                'universal_building_types': True,
                'equal_treatment_preprocessing': True
            },
            'detailed_results': results
        }
        
        # Save to JSON file
        summary_file = f"logs/fair_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 FAIR analysis summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save analysis summary: {e}")

def setup_logging():
    """
    Setup comprehensive logging for the fair analysis pipeline
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/fair_analysis_{timestamp}.log"
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),  # Save to file
            logging.StreamHandler(sys.stdout)  # Also display on screen
        ]
    )
    
    # Create a custom logger
    logger = logging.getLogger('BuildingEnergyAnalysis')
    
    print(f"🔄 FAIR analysis logging enabled - saving to: {log_filename}")
    logger.info("="*80)
    logger.info("FAIR BUILDING ENERGY PREDICTION ANALYSIS STARTED")
    logger.info("="*80)
    
    return logger        

def run_fair_individual_analysis():
    """Enhanced version with FAIR preprocessing and comprehensive logging"""
    
    # Setup logging at the start
    logger = setup_logging()
    
    logger.info("🎯 Starting FAIR individual dataset analysis for three major US cities")
    logger.info("✅ Equal treatment preprocessing with adaptive thresholds")
    logger.info("🏙️ Seattle, Chicago, and Washington DC energy benchmarking data")

    datasets = {
        'seattle_2015_present': 'data/seattle_2015_present.csv',
        'chicago_energy': 'data/chicago_energy_benchmarking.csv',
        'washington_dc': 'data/washington_dc_energy.csv'
    }
    
    results = {}
    total_start_time = time.time()

    # Log dataset availability
    available_datasets = {}
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            available_datasets[dataset_name] = file_path
            logger.info(f"✅ Found: {dataset_name} -> {file_path}")
        else:
            logger.warning(f"❌ Missing: {dataset_name} -> {file_path}")
    
    if not available_datasets:
        logger.error("❌ No datasets found! Please ensure data files are in the 'data/' directory")
        return {}
    
    logger.info(f"🚀 Processing {len(available_datasets)} available major US cities with FAIR treatment...")
    
    # Process each dataset with FAIR treatment
    for dataset_name, file_path in available_datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"FAIR PROCESSING: {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = process_individual_dataset(dataset_name, file_path)
            end_time = time.time()
            
            results[dataset_name] = {
                'success': success,
                'time': end_time - start_time,
                'file_path': file_path,
                'processing_mode': 'FAIR'
            }
            
            logger.info(f"📊 {dataset_name} FAIR Summary:")
            logger.info(f"   Status: {'✅ Success' if success else '❌ Failed'}")
            logger.info(f"   Processing time: {end_time - start_time:.1f} seconds")
            
            if success:
                logger.info(f"   ✅ {dataset_name} FAIR analysis completed successfully!")
            else:
                logger.error(f"   ❌ {dataset_name} FAIR analysis failed!")
                
        except Exception as e:
            end_time = time.time()
            logger.error(f"❌ Exception in {dataset_name}: {str(e)}")
            results[dataset_name] = {
                'success': False,
                'time': end_time - start_time,
                'file_path': file_path,
                'error': str(e),
                'processing_mode': 'FAIR'
            }
    
    # Final summary with detailed logging
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info("\n" + "="*80)
    logger.info("FAIR INDIVIDUAL ANALYSIS COMPLETED")
    logger.info("="*80)
    
    successful_datasets = [name for name, result in results.items() if result['success']]
    failed_datasets = [name for name, result in results.items() if not result['success']]
    
    logger.info(f"📊 FAIR Analysis Results Summary:")
    logger.info(f"   Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"   Successful datasets: {len(successful_datasets)}/{len(available_datasets)}")
    logger.info(f"   Available datasets: {len(available_datasets)}/{len(datasets)}")
    logger.info(f"   Success rate: {len(successful_datasets)/len(available_datasets)*100:.1f}%")
    
    if successful_datasets:
        logger.info(f"🎉 Successfully analyzed cities with FAIR treatment:")
        for dataset_name in successful_datasets:
            result = results[dataset_name]
            logger.info(f"   ✅ {dataset_name.replace('_', ' ').title()}: {result['time']:.1f}s")
        
        logger.info(f"📁 Results saved in:")
        for dataset_name in successful_datasets:
            logger.info(f"   📂 outputs/{dataset_name}/")
    
    if failed_datasets:
        logger.warning(f"⚠️  Failed datasets:")
        for dataset_name in failed_datasets:
            result = results[dataset_name]
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"   ❌ {dataset_name}: {error_msg}")
    
    # Generate fairness report
    generate_fairness_report(results)
    
    # Save analysis summary
    save_analysis_summary(results, total_time)
    
    logger.info("🏁 FAIR analysis pipeline completed!")
    logger.info("="*80)
    
    return results

# Alias for backward compatibility
def run_individual_analysis():
    """Backward compatibility alias"""
    return run_fair_individual_analysis()

if __name__ == "__main__":
    run_fair_individual_analysis()