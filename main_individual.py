"""
Individual Dataset Analysis for Building Energy Prediction
针对每个数据集进行独立分析，避免数据不兼容问题 - FIXED VERSION
"""
import logging
from datetime import datetime
import os
import sys
import time
import pandas as pd
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
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        create_individual_directories(dataset_name)
        
        print("🔄 Step 1: Data Preprocessing...")

        try:
            from preprocessing.clean_data import preprocess_data
        except ImportError as e:
            print(f"❌ Cannot import preprocessing module: {e}")
            return False
        
        result = preprocess_data(file_path)
        if result is None:
            print(f"❌ Preprocessing failed for {dataset_name}")
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
        
        print(f"✅ Preprocessing completed for {dataset_name}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        print("\n🤖 Step 2: Model Training...")
        train_models_for_dataset(dataset_name)

        print("\n📊 Step 3: Model Evaluation...")
        evaluate_models_for_dataset(dataset_name)

        if len(X_train) > 500:
            print("\n🎯 Step 4: Classification Models...")
            train_classification_for_dataset(dataset_name, X_train, y_train)
        else:
            print(f"\n⚠️  Skipping classification for {dataset_name} (insufficient data: {len(X_train)} samples)")
        
        print(f"✅ {dataset_name} analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_models_for_dataset(dataset_name):
    """为单个数据集训练模型 - FIXED VERSION"""
    
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

    try:
        print(f"  🎯 Training classification models for {dataset_name}...")

        import numpy as np

        def create_simple_labels(energy_values):
            """创建简单的能效标签"""
            q1 = energy_values.quantile(0.25)
            q2 = energy_values.quantile(0.50)
            q3 = energy_values.quantile(0.75)
            
            def assign_label(value):
                if pd.isna(value):
                    return 'Unknown'
                elif value <= q1:
                    return 'Excellent' 
                elif value <= q2:
                    return 'Good'
                elif value <= q3:
                    return 'Average'
                else:
                    return 'Poor' 
            
            return energy_values.apply(assign_label)
        
        labels = create_simple_labels(y_train)

        output_dir = f"outputs/{dataset_name}"
        X_train.to_csv(f"{output_dir}/unified_features.csv", index=False)
        labels.to_csv(f"{output_dir}/unified_labels.csv", index=False)
        
        print(f"     ✓ Classification labels created: {labels.value_counts().to_dict()}")

        try:
            from models.train_classification_individual import main as train_class
            train_class(dataset_name)
            print("     ✅ Classification training completed")
        except Exception as e:
            print(f"     ❌ Classification training failed: {e}")
        
    except Exception as e:
        print(f"     ❌ Classification setup error: {e}")


def generate_fairness_report(results):
    """
    Generate a fairness analysis report comparing three cities
    """
    print("\n" + "="*80)
    print("FAIRNESS ANALYSIS REPORT")
    print("="*80)
    
    print("City-Specific Performance Analysis:")
    
    for dataset_name, result in results.items():
        if result['success']:
            print(f"\n✅ {dataset_name.upper()}:")
            print(f"   Status: Successfully analyzed")
            print(f"   Processing time: {result['time']:.1f} seconds")
            
            # Try to read evaluation results
            try:
                eval_file = f"outputs/{dataset_name}/model_evaluation_results.csv"
                if os.path.exists(eval_file):
                    df = pd.read_csv(eval_file)
                    best_model = df.loc[df['R2'].idxmax()]
                    print(f"   Best model: {best_model['Model']}")
                    print(f"   Best R²: {best_model['R2']:.4f}")
                    print(f"   Best RMSE: {best_model['RMSE']:.2f}")
            except Exception as e:
                print(f"   Could not load detailed results: {e}")
        else:
            print(f"\n❌ {dataset_name.upper()}:")
            print(f"   Status: Analysis failed")
    
    print(f"\n📊 FAIRNESS ASSESSMENT:")
    print(f"Each city has been processed with:")
    print(f"   ✅ City-specific target variable detection")
    print(f"   ✅ City-specific data cleaning thresholds")
    print(f"   ✅ City-specific building type standardization")
    print(f"   ✅ City-specific outlier removal strategies")
    
    return results

def save_analysis_summary(results, total_time):
    """Save a detailed analysis summary"""
    
    logger = logging.getLogger('BuildingEnergyAnalysis')
    
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'datasets_processed': len(results),
            'successful_analyses': sum(1 for r in results.values() if r['success']),
            'failed_analyses': sum(1 for r in results.values() if not r['success']),
            'detailed_results': results
        }
        
        # Save to JSON file
        summary_file = f"logs/analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Analysis summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save analysis summary: {e}")

def setup_logging():
    """
    Setup comprehensive logging for the analysis pipeline
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/analysis_{timestamp}.log"
    
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
    
    print(f"🔄 Logging enabled - saving to: {log_filename}")
    logger.info("="*80)
    logger.info("BUILDING ENERGY PREDICTION ANALYSIS STARTED")
    logger.info("="*80)
    
    return logger        

def run_individual_analysis():
    """Enhanced version with comprehensive logging"""
    
    # Setup logging at the start
    logger = setup_logging()
    
    logger.info("🎯 Starting individual dataset analysis for three major US cities")
    logger.info("✅ Seattle, Chicago, and Washington DC energy benchmarking data")

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
    
    logger.info(f"🚀 Processing {len(available_datasets)} available major US cities...")
    
    # Process each dataset with detailed logging
    for dataset_name, file_path in available_datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING {dataset_name.upper()} DATASET")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            success = process_individual_dataset(dataset_name, file_path)
            end_time = time.time()
            
            results[dataset_name] = {
                'success': success,
                'time': end_time - start_time,
                'file_path': file_path
            }
            
            logger.info(f"📊 {dataset_name} Summary:")
            logger.info(f"   Status: {'✅ Success' if success else '❌ Failed'}")
            logger.info(f"   Processing time: {end_time - start_time:.1f} seconds")
            
            if success:
                logger.info(f"   ✅ {dataset_name} analysis completed successfully!")
            else:
                logger.error(f"   ❌ {dataset_name} analysis failed!")
                
        except Exception as e:
            end_time = time.time()
            logger.error(f"❌ Exception in {dataset_name}: {str(e)}")
            results[dataset_name] = {
                'success': False,
                'time': end_time - start_time,
                'file_path': file_path,
                'error': str(e)
            }
    
    # Final summary with detailed logging
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info("\n" + "="*80)
    logger.info("INDIVIDUAL ANALYSIS COMPLETED")
    logger.info("="*80)
    
    successful_datasets = [name for name, result in results.items() if result['success']]
    failed_datasets = [name for name, result in results.items() if not result['success']]
    
    logger.info(f"📊 Final Results Summary:")
    logger.info(f"   Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"   Successful datasets: {len(successful_datasets)}/{len(available_datasets)}")
    logger.info(f"   Available datasets: {len(available_datasets)}/{len(datasets)}")
    
    if successful_datasets:
        logger.info(f"🎉 Successfully analyzed cities:")
        for dataset_name in successful_datasets:
            result = results[dataset_name]
            logger.info(f"   ✅ {dataset_name}: {result['time']:.1f}s")
        
        logger.info(f"📁 Results saved in:")
        for dataset_name in successful_datasets:
            logger.info(f"   📂 outputs/{dataset_name}/")
    
    if failed_datasets:
        logger.warning(f"⚠️  Failed datasets:")
        for dataset_name in failed_datasets:
            result = results[dataset_name]
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"   ❌ {dataset_name}: {error_msg}")
    
    # Save analysis summary
    save_analysis_summary(results, total_time)
    
    logger.info("🏁 Analysis pipeline completed!")
    logger.info("="*80)
    
    return results



if __name__ == "__main__":
    run_individual_analysis()