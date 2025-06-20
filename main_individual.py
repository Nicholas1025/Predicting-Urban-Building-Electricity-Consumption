"""
Individual Dataset Analysis for Building Energy Prediction
针对每个数据集进行独立分析，避免数据不兼容问题 - FIXED VERSION
"""

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

def run_individual_analysis():

    print("="*80)
    print("INDIVIDUAL DATASET ANALYSIS PIPELINE")
    print("="*80)
    print("🎯 Analyzing each dataset independently to avoid compatibility issues")
    print("✅ This approach eliminates cross-year negative R² problems")

    datasets = {
        'seattle_2015': 'data/2015-building-energy-benchmarking.csv',
        'seattle_2016': 'data/2016-building-energy-benchmarking.csv',
        'nyc_2021': 'data/energy_disclosure_2021_rows.csv'
    }
    
    results = {}
    total_start_time = time.time()

    available_datasets = {}
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            available_datasets[dataset_name] = file_path
            print(f"✅ Found: {dataset_name} -> {file_path}")
        else:
            print(f"❌ Missing: {dataset_name} -> {file_path}")
    
    if not available_datasets:
        print("❌ No datasets found! Please ensure data files are in the 'data/' directory")
        return {}
    
    print(f"\n🚀 Processing {len(available_datasets)} available datasets...")
    
    for dataset_name, file_path in available_datasets.items():
        start_time = time.time()
        success = process_individual_dataset(dataset_name, file_path)
        end_time = time.time()
        
        results[dataset_name] = {
            'success': success,
            'time': end_time - start_time,
            'file_path': file_path
        }
        
        print(f"\n📊 {dataset_name} Summary:")
        print(f"   Status: {'✅ Success' if success else '❌ Failed'}")
        print(f"   Time: {end_time - start_time:.1f} seconds")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    print("\n" + "="*80)
    print("INDIVIDUAL ANALYSIS COMPLETED")
    print("="*80)
    
    successful_datasets = [name for name, result in results.items() if result['success']]
    failed_datasets = [name for name, result in results.items() if not result['success']]
    
    print(f"📊 Results Summary:")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Successful datasets: {len(successful_datasets)}/{len(available_datasets)}")
    print(f"   Available datasets: {len(available_datasets)}/{len(datasets)}")
    
    if successful_datasets:
        print(f"\n🎉 Successfully analyzed:")
        for dataset_name in successful_datasets:
            result = results[dataset_name]
            print(f"   ✅ {dataset_name}: {result['time']:.1f}s")
        
        print(f"\n📁 Results saved in:")
        for dataset_name in successful_datasets:
            print(f"   📂 outputs/{dataset_name}/")
        
        print(f"\n🌐 Start the web dashboard to view detailed results!")
        print(f"   Command: python run_project.py --dashboard")
    
    if failed_datasets:
        print(f"\n⚠️  Failed datasets:")
        for dataset_name in failed_datasets:
            result = results[dataset_name]
            print(f"   ❌ {dataset_name}: Check error messages above")
    
    if not successful_datasets:
        print(f"\n❌ No datasets were successfully analyzed")
        print(f"Please check:")
        print(f"   📁 Data file paths and formats")
        print(f"   🐍 Python dependencies (run system check)")
        print(f"   📋 Error messages above for specific issues")
    
    return results


if __name__ == "__main__":
    run_individual_analysis()