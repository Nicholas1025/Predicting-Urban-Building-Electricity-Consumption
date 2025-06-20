"""
Classification Diagnosis Script
Check why classification models are not being generated
"""

import pandas as pd
import numpy as np
import os
import sys

def check_classification_data(dataset_name):
    """Check if classification data and models exist for a dataset"""
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION DIAGNOSIS: {dataset_name.upper()}")
    print('='*60)
    
    base_dir = f"outputs/{dataset_name}"
    
    # Check if unified files exist
    unified_features = f"{base_dir}/unified_features.csv"
    unified_labels = f"{base_dir}/unified_labels.csv"
    
    print(f"🔍 Checking classification input files:")
    print(f"   Features file: {unified_features}")
    print(f"   Exists: {'✅' if os.path.exists(unified_features) else '❌'}")
    print(f"   Labels file: {unified_labels}")
    print(f"   Exists: {'✅' if os.path.exists(unified_labels) else '❌'}")
    
    if not os.path.exists(unified_features) or not os.path.exists(unified_labels):
        print("❌ Classification input files missing!")
        
        # Try to create them from existing data
        print("🔧 Attempting to create classification data from existing regression data...")
        
        try:
            # Load regression data
            X_train = pd.read_csv(f"{base_dir}/X_train.csv")
            y_train = pd.read_csv(f"{base_dir}/y_train.csv").iloc[:, 0]
            
            print(f"✅ Loaded regression data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            # Create classification labels
            def create_classification_labels(energy_values):
                """Create balanced classification labels"""
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
            
            labels = create_classification_labels(y_train)
            label_counts = labels.value_counts()
            
            print(f"📊 Classification label distribution:")
            for label, count in label_counts.items():
                print(f"   {label}: {count} samples")
            
            # Save classification data
            X_train.to_csv(unified_features, index=False)
            labels.to_csv(unified_labels, index=False)
            
            print(f"✅ Created classification input files")
            
        except Exception as e:
            print(f"❌ Failed to create classification data: {e}")
            return False
    
    else:
        # Files exist, check their content
        try:
            X = pd.read_csv(unified_features)
            y = pd.read_csv(unified_labels).iloc[:, 0]
            
            print(f"✅ Classification data loaded:")
            print(f"   Features: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"   Labels: {len(y)} samples")
            
            # Check label distribution
            if hasattr(y, 'value_counts'):
                label_counts = y.value_counts()
                print(f"📊 Label distribution:")
                for label, count in label_counts.items():
                    print(f"   {label}: {count} samples")
                
                # Check if distribution is suitable for classification
                min_samples = label_counts.min()
                num_classes = len(label_counts)
                
                print(f"📈 Classification viability:")
                print(f"   Number of classes: {num_classes}")
                print(f"   Minimum class size: {min_samples}")
                print(f"   Total samples: {len(y)}")
                
                if num_classes < 2:
                    print("❌ Not enough classes for classification")
                    return False
                elif min_samples < 5:
                    print("⚠️  Some classes have very few samples")
                elif len(y) < 50:
                    print("⚠️  Very few total samples for classification")
                else:
                    print("✅ Data looks suitable for classification")
            
        except Exception as e:
            print(f"❌ Error reading classification data: {e}")
            return False
    
    # Check if classification models exist
    models_dir = f"{base_dir}/models"
    classification_models = [
        "model_random_forest_classifier.pkl",
        "model_xgboost_classifier.pkl", 
        "model_svm_classifier.pkl"
    ]
    
    print(f"\n🤖 Checking classification models:")
    models_exist = 0
    for model_file in classification_models:
        model_path = f"{models_dir}/{model_file}"
        exists = os.path.exists(model_path)
        print(f"   {model_file}: {'✅' if exists else '❌'}")
        if exists:
            models_exist += 1
    
    # Check classification charts
    charts_dir = f"{base_dir}/charts"
    classification_charts = [
        "confusion_matrices_classification.png",
        "classification_metrics_comparison.png"
    ]
    
    print(f"\n📊 Checking classification charts:")
    charts_exist = 0
    for chart_file in classification_charts:
        chart_path = f"{charts_dir}/{chart_file}"
        exists = os.path.exists(chart_path)
        print(f"   {chart_file}: {'✅' if exists else '❌'}")
        if exists:
            charts_exist += 1
    
    # Check classification results table
    tables_dir = f"{base_dir}/tables"
    classification_table = f"{tables_dir}/classification_performance.csv"
    table_exists = os.path.exists(classification_table)
    print(f"\n📋 Classification results table:")
    print(f"   classification_performance.csv: {'✅' if table_exists else '❌'}")
    
    # Summary
    print(f"\n📊 CLASSIFICATION STATUS SUMMARY:")
    print(f"   Input files: {'✅' if os.path.exists(unified_features) and os.path.exists(unified_labels) else '❌'}")
    print(f"   Models: {models_exist}/3 exist")
    print(f"   Charts: {charts_exist}/2 exist")
    print(f"   Results table: {'✅' if table_exists else '❌'}")
    
    if models_exist == 0:
        print("🔧 RECOMMENDATION: Run classification training")
        print(f"   Command: python -c \"from models.train_classification_individual import main; main('{dataset_name}')\"")
        return False
    elif charts_exist == 0:
        print("🔧 RECOMMENDATION: Classification models exist but charts missing")
        return False
    else:
        print("✅ Classification appears complete")
        return True


def run_classification_training(dataset_name):
    """Attempt to run classification training for a dataset"""
    
    print(f"\n🚀 Running classification training for {dataset_name}...")
    
    try:
        # Import and run classification training
        sys.path.append('.')
        from models.train_classification_individual import main as train_classification
        
        result = train_classification(dataset_name)
        
        if result:
            print(f"✅ Classification training completed for {dataset_name}")
            return True
        else:
            print(f"❌ Classification training failed for {dataset_name}")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import classification training module: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during classification training: {e}")
        return False


def main():
    """Main diagnosis function"""
    
    print("🔍 CLASSIFICATION DIAGNOSIS TOOL")
    print("Checking why classification models are missing...")
    
    datasets = ['seattle_2015_present', 'chicago_energy', 'washington_dc']
    
    for dataset in datasets:
        base_dir = f"outputs/{dataset}"
        if os.path.exists(base_dir):
            classification_ok = check_classification_data(dataset)
            
            if not classification_ok:
                print(f"\n🔧 Attempting to fix classification for {dataset}...")
                success = run_classification_training(dataset)
                
                if success:
                    print(f"✅ Fixed classification for {dataset}")
                else:
                    print(f"❌ Could not fix classification for {dataset}")
            
        else:
            print(f"\n❌ Dataset directory not found: {base_dir}")
    
    print(f"\n🏁 Diagnosis completed!")
    print("💡 If classification training failed, you may need to:")
    print("   1. Check if samples are sufficient (>50 total, >5 per class)")
    print("   2. Manually run: python models/train_classification_individual.py [dataset_name]")
    print("   3. Check error logs for specific issues")


if __name__ == "__main__":
    main()