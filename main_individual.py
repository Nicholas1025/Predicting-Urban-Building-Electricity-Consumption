"""
Individual Dataset Analysis for Building Energy Prediction
针对每个数据集进行独立分析，避免数据不兼容问题
"""

import os
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append('.')

def create_individual_directories(dataset_name):
    """为每个数据集创建独立的输出目录"""
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
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # 创建输出目录
        create_individual_directories(dataset_name)
        
        # 设置环境变量，让子模块知道当前处理的数据集
        os.environ['CURRENT_DATASET'] = dataset_name
        os.environ['OUTPUT_PREFIX'] = f"outputs/{dataset_name}"
        
        # Step 1: 数据预处理
        print("🔄 Step 1: Data Preprocessing...")
        from preprocessing.clean_data import preprocess_data
        
        result = preprocess_data(file_path)
        if result is None:
            print(f"❌ Preprocessing failed for {dataset_name}")
            return False
            
        X_train, X_test, y_train, y_test, scaler, feature_names = result
        
        # 保存预处理结果到独立文件夹
        output_dir = f"outputs/{dataset_name}"
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # 保存特征名和scaler
        with open(f"{output_dir}/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        
        print(f"✅ Preprocessing completed for {dataset_name}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Step 2: 模型训练
        print("\n🤖 Step 2: Model Training...")
        train_models_for_dataset(dataset_name)
        
        # Step 3: 模型评估
        print("\n📊 Step 3: Model Evaluation...")
        evaluate_models_for_dataset(dataset_name)
        
        # Step 4: 分类模型（如果数据足够）
        if len(X_train) > 500:  # 只有足够数据才做分类
            print("\n🎯 Step 4: Classification Models...")
            train_classification_for_dataset(dataset_name, X_train, y_train)
        
        print(f"✅ {dataset_name} analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理环境变量
        os.environ.pop('CURRENT_DATASET', None)
        os.environ.pop('OUTPUT_PREFIX', None)

def train_models_for_dataset(dataset_name):
    """为单个数据集训练模型"""
    
    # 修改工作目录，让现有的训练脚本使用正确的路径
    original_dir = os.getcwd()
    output_dir = f"outputs/{dataset_name}"
    
    try:
        # 临时修改输出路径环境变量
        os.environ['MODEL_OUTPUT_DIR'] = f"{output_dir}/models"
        os.environ['CHART_OUTPUT_DIR'] = f"{output_dir}/charts" 
        os.environ['PRED_OUTPUT_DIR'] = output_dir
        
        # 训练XGBoost
        print("  🚀 Training XGBoost...")
        from models.train_xgboost_individual import main as train_xgb
        train_xgb(dataset_name)
        
        # 训练Random Forest
        print("  🌲 Training Random Forest...")
        from models.train_rf_individual import main as train_rf
        train_rf(dataset_name)
        
        # 训练SVR
        print("  ⚡ Training SVR...")
        from models.train_svr_individual import main as train_svr
        train_svr(dataset_name)
        
    except Exception as e:
        print(f"❌ Model training error: {e}")
    finally:
        # 恢复环境变量
        os.environ.pop('MODEL_OUTPUT_DIR', None)
        os.environ.pop('CHART_OUTPUT_DIR', None) 
        os.environ.pop('PRED_OUTPUT_DIR', None)

def evaluate_models_for_dataset(dataset_name):
    """评估单个数据集的模型"""
    try:
        os.environ['EVAL_OUTPUT_DIR'] = f"outputs/{dataset_name}"
        from evaluation.evaluate_models_individual import main as eval_models
        eval_models(dataset_name)
    except Exception as e:
        print(f"⚠️  Evaluation error: {e}")
    finally:
        os.environ.pop('EVAL_OUTPUT_DIR', None)

def train_classification_for_dataset(dataset_name, X_train, y_train):
    """为单个数据集训练分类模型"""
    try:
        # 创建分类标签
        from preprocessing.multi_dataset_processor import create_energy_efficiency_labels
        labels = create_energy_efficiency_labels(y_train)
        
        # 保存分类数据
        output_dir = f"outputs/{dataset_name}"
        X_train.to_csv(f"{output_dir}/unified_features.csv", index=False)
        labels.to_csv(f"{output_dir}/unified_labels.csv", index=False)
        
        # 训练分类模型
        os.environ['CLASS_OUTPUT_DIR'] = output_dir
        from models.train_classification_individual import main as train_class
        train_class(dataset_name)
        
    except Exception as e:
        print(f"⚠️  Classification training error: {e}")
    finally:
        os.environ.pop('CLASS_OUTPUT_DIR', None)

def run_individual_analysis():
    """运行独立数据集分析的主函数"""
    print("="*80)
    print("INDIVIDUAL DATASET ANALYSIS PIPELINE")
    print("="*80)
    print("🎯 Analyzing each dataset independently to avoid compatibility issues")
    
    # 定义数据集
    datasets = {
        'seattle_2015': 'data/2015-building-energy-benchmarking.csv',
        'seattle_2016': 'data/2016-building-energy-benchmarking.csv',
        'nyc_2021': 'data/energy_disclosure_2021_rows.csv'
    }
    
    results = {}
    total_start_time = time.time()
    
    for dataset_name, file_path in datasets.items():
        start_time = time.time()
        success = process_individual_dataset(dataset_name, file_path)
        end_time = time.time()
        
        results[dataset_name] = {
            'success': success,
            'time': end_time - start_time
        }
        
        print(f"\n📊 {dataset_name} Summary:")
        print(f"   Status: {'✅ Success' if success else '❌ Failed'}")
        print(f"   Time: {end_time - start_time:.1f} seconds")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 最终总结
    print("\n" + "="*80)
    print("INDIVIDUAL ANALYSIS COMPLETED")
    print("="*80)
    
    successful_datasets = [name for name, result in results.items() if result['success']]
    
    print(f"📊 Results Summary:")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Successful datasets: {len(successful_datasets)}/{len(datasets)}")
    
    for dataset_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"   {status} {dataset_name}: {result['time']:.1f}s")
    
    if successful_datasets:
        print(f"\n🎉 Successfully analyzed: {', '.join(successful_datasets)}")
        print(f"📁 Results saved in: outputs/[dataset_name]/")
        print(f"🌐 Start dashboard to view results!")
    else:
        print(f"\n⚠️  No datasets were successfully analyzed")
        print(f"Please check the data files and error messages above")
    
    return results

if __name__ == "__main__":
    run_individual_analysis()