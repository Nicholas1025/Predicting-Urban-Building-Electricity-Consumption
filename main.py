"""
Enhanced Main Machine Learning Pipeline for Building Energy Consumption Prediction
Supports multiple datasets: Seattle 2015, Seattle 2016, and NYC 2021
Orchestrates data preprocessing, model training, and evaluation
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
from preprocessing.clean_data import preprocess_multiple_datasets
from evaluation.evaluate_models import (
    load_true_values, load_model_predictions, evaluate_all_models,
    plot_predicted_vs_actual, plot_model_comparison, plot_residuals_analysis,
    save_results_summary
)

warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('.')


def print_header(title, char="=", width=80):
    """Print formatted header"""
    print("\n" + char * width)
    print(f"{title.upper().center(width)}")
    print(char * width)


def print_step(step_num, title, char="-", width=80):
    """Print formatted step header"""
    step_title = f"STEP {step_num}: {title}"
    print("\n" + char * width)
    print(f"{step_title.center(width)}")
    print(char * width)


def create_directories():
    """Create necessary directories"""
    directories = [
        "outputs",
        "outputs/charts",
        "models",
        "data"  # Ensure data directory exists
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")


def check_datasets():
    """
    Check which datasets are available
    
    Returns:
        list: Available dataset file paths
    """
    dataset_paths = [
        "data/2016-building-energy-benchmarking.csv",
        "data/2015-building-energy-benchmarking.csv", 
        "data/energy_disclosure_2021_rows.csv"
    ]
    
    available_datasets = []
    missing_datasets = []
    
    print("📋 Checking dataset availability:")
    
    for path in dataset_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # Size in MB
            available_datasets.append(path)
            dataset_name = os.path.basename(path)
            print(f"  ✅ {dataset_name} ({file_size:.1f} MB)")
        else:
            missing_datasets.append(path)
            dataset_name = os.path.basename(path)
            print(f"  ❌ {dataset_name} - NOT FOUND")
    
    if available_datasets:
        print(f"\n📊 Found {len(available_datasets)} available dataset(s)")
        if missing_datasets:
            print(f"⚠️  {len(missing_datasets)} dataset(s) missing but will continue with available data")
    else:
        print("❌ No datasets found!")
    
    return available_datasets


def train_xgboost_model():
    """Train XGBoost model"""
    print("🚀 Training XGBoost model...")
    
    try:
        # Try to import from existing script
        from models.train_xgboost import main as train_xgb_main
        train_xgb_main()
    except ImportError:
        # Fallback: Create XGBoost model inline
        print("  train_xgboost.py not found. Training XGBoost model inline...")
        
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        
        # Load preprocessed data
        X_train = pd.read_csv("outputs/X_train.csv")
        X_test = pd.read_csv("outputs/X_test.csv")
        y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
        # Create and train model
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        print("  Training XGBoost...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ XGBoost Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Save model and predictions
        joblib.dump(model, "outputs/model_xgb.pkl")
        pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_xgb.csv", index=False)
        
        print("  💾 XGBoost model saved")


def train_random_forest_model():
    """Train Random Forest model"""
    print("🌲 Training Random Forest model...")
    
    try:
        # Try to import from existing script
        from models.train_rf import main as train_rf_main
        train_rf_main()
    except ImportError:
        # Fallback: Create Random Forest model inline
        print("  train_rf.py not found. Training Random Forest model inline...")
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        
        # Load preprocessed data
        X_train = pd.read_csv("outputs/X_train.csv")
        X_test = pd.read_csv("outputs/X_test.csv")
        y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        print("  Training Random Forest...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ Random Forest Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Save model and predictions
        joblib.dump(model, "outputs/model_rf.pkl")
        pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_rf.csv", index=False)
        
        print("  💾 Random Forest model saved")


def train_gradient_boosting_model():
    """Train Gradient Boosting model"""
    print("📈 Training Gradient Boosting model...")
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        
        # Load preprocessed data
        X_train = pd.read_csv("outputs/X_train.csv")
        X_test = pd.read_csv("outputs/X_test.csv")
        y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
        # Create and train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        print("  Training Gradient Boosting...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ Gradient Boosting Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Save model and predictions
        joblib.dump(model, "outputs/model_gb.pkl")
        pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_gb.csv", index=False)
        
        print("  💾 Gradient Boosting model saved")
        
    except Exception as e:
        print(f"  ❌ Gradient Boosting training failed: {e}")


def train_svr_model():
    """Train Support Vector Regression model"""
    print("⚡ Training Support Vector Regression model...")
    
    try:
        from sklearn.svm import SVR
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import joblib
        
        # Load preprocessed data
        X_train = pd.read_csv("outputs/X_train.csv")
        X_test = pd.read_csv("outputs/X_test.csv")
        y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
        # For large datasets, use a subset for SVR training
        if len(X_train) > 5000:
            print("  Using subset of data for SVR (computational efficiency)")
            subset_indices = np.random.choice(len(X_train), 5000, replace=False)
            X_train_subset = X_train.iloc[subset_indices]
            y_train_subset = y_train.iloc[subset_indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        # Create and train model
        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1
        )
        
        print("  Training SVR...")
        model.fit(X_train_subset, y_train_subset)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  ✅ SVR Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Save model and predictions
        joblib.dump(model, "outputs/model_svr.pkl")
        pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_svr.csv", index=False)
        
        print("  💾 SVR model saved")
        
    except Exception as e:
        print(f"  ❌ SVR training failed: {e}")


def run_data_preprocessing(available_datasets):
    """
    Step 1: Enhanced Data Preprocessing for Multiple Datasets
    """
    print_step(1, "MULTI-DATASET PREPROCESSING")
    
    if not available_datasets:
        print("❌ No datasets available for preprocessing")
        return False
    
    print(f"📊 Processing {len(available_datasets)} dataset(s):")
    for i, path in enumerate(available_datasets, 1):
        dataset_name = os.path.basename(path)
        print(f"  {i}. {dataset_name}")
    
    try:
        # Run enhanced preprocessing with multiple datasets
        print("\n🔄 Running multi-dataset preprocessing...")
        result = preprocess_multiple_datasets(
            file_paths=available_datasets,
            test_size=0.2,
            random_state=42
        )
        
        if result is not None:
            X_train, X_test, y_train, y_test, scaler, feature_names, target_name = result
            
            # Save preprocessed data
            print("\n💾 Saving preprocessed data...")
            X_train.to_csv("outputs/X_train.csv", index=False)
            X_test.to_csv("outputs/X_test.csv", index=False)
            y_train.to_csv("outputs/y_train.csv", index=False)
            y_test.to_csv("outputs/y_test.csv", index=False)
            
            # Save feature names
            with open("outputs/feature_names.txt", "w") as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            
            # Save target information
            with open("outputs/target_info.txt", "w") as f:
                f.write(f"Target variable: {target_name}\n")
                f.write(f"Training samples: {len(y_train)}\n")
                f.write(f"Test samples: {len(y_test)}\n")
                f.write(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]\n")
                f.write(f"Datasets used: {len(available_datasets)}\n")
                for dataset in available_datasets:
                    f.write(f"  - {os.path.basename(dataset)}\n")
            
            # Save scaler
            import joblib
            joblib.dump(scaler, "outputs/scaler.pkl")
            
            print("✅ Multi-dataset preprocessing completed successfully!")
            print(f"   📋 Combined dataset size: {len(X_train) + len(X_test)} buildings")
            print(f"   🎯 Target variable: {target_name}")
            print(f"   📊 Training samples: {X_train.shape[0]}")
            print(f"   🧪 Test samples: {X_test.shape[0]}")
            print(f"   🔢 Features: {X_train.shape[1]}")
            print(f"   🗂️ Datasets combined: {len(available_datasets)}")
            
            return True
        else:
            print("❌ Multi-dataset preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during multi-dataset preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_training():
    """
    Step 2: Enhanced Model Training
    """
    print_step(2, "ENHANCED MODEL TRAINING")
    
    # Check if preprocessed data exists
    required_files = ["outputs/X_train.csv", "outputs/X_test.csv", 
                     "outputs/y_train.csv", "outputs/y_test.csv"]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing preprocessed data files: {missing_files}")
        print("Please run data preprocessing first.")
        return False
    
    models_trained = []
    
    # Train XGBoost
    try:
        print("\n" + "="*60)
        start_time = time.time()
        train_xgboost_model()
        end_time = time.time()
        models_trained.append(f"XGBoost ({end_time - start_time:.1f}s)")
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
    
    # Train Random Forest
    try:
        print("\n" + "="*60)
        start_time = time.time()
        train_random_forest_model()
        end_time = time.time()
        models_trained.append(f"Random Forest ({end_time - start_time:.1f}s)")
    except Exception as e:
        print(f"❌ Random Forest training failed: {e}")
    
    # Train Gradient Boosting
    try:
        print("\n" + "="*60)
        start_time = time.time()
        train_gradient_boosting_model()
        end_time = time.time()
        models_trained.append(f"Gradient Boosting ({end_time - start_time:.1f}s)")
    except Exception as e:
        print(f"❌ Gradient Boosting training failed: {e}")
    
    # Train SVR (optional for large datasets)
    data_size = len(pd.read_csv("outputs/X_train.csv"))
    if data_size <= 10000:  # Only train SVR for smaller datasets
        try:
            print("\n" + "="*60)
            start_time = time.time()
            train_svr_model()
            end_time = time.time()
            models_trained.append(f"SVR ({end_time - start_time:.1f}s)")
        except Exception as e:
            print(f"❌ SVR training failed: {e}")
    else:
        print(f"\n⏭️  Skipping SVR training (dataset too large: {data_size} samples)")
    
    if models_trained:
        print(f"\n✅ Model training completed!")
        print(f"   🤖 Models trained: {len(models_trained)}")
        for model in models_trained:
            print(f"   • {model}")
        return True
    else:
        print("❌ No models were trained successfully!")
        return False


def run_model_evaluation():
    """
    Step 3: Enhanced Model Evaluation
    """
    print_step(3, "COMPREHENSIVE MODEL EVALUATION")
    
    try:
        # Load true values
        print("📊 Loading test data and predictions...")
        y_true = load_true_values()
        if y_true is None:
            print("❌ Could not load true values")
            return False
        
        # Load model predictions
        predictions_dict = load_model_predictions()
        if not predictions_dict:
            print("❌ No model predictions found")
            return False
        
        print(f"✅ Found predictions for {len(predictions_dict)} models:")
        for model_name in predictions_dict.keys():
            print(f"   • {model_name}")
        
        # Load target information
        target_info = "Unknown"
        if os.path.exists("outputs/target_info.txt"):
            with open("outputs/target_info.txt", "r") as f:
                lines = f.readlines()
                if lines:
                    target_info = lines[0].replace("Target variable: ", "").strip()
        
        print(f"🎯 Target variable: {target_info}")
        
        # Evaluate all models
        print("\n📈 Evaluating model performance...")
        results_df = evaluate_all_models(y_true, predictions_dict)
        
        # Create visualizations
        print("\n🎨 Creating comprehensive visualizations...")
        plot_predicted_vs_actual(y_true, predictions_dict)
        plot_model_comparison(results_df)
        plot_residuals_analysis(y_true, predictions_dict)
        
        # Save results
        print("\n💾 Saving evaluation results...")
        save_results_summary(results_df)
        
        print("✅ Model evaluation completed successfully!")
        
        # Print detailed summary
        print(f"\n{'='*80}")
        print("🏆 MODEL PERFORMANCE RANKING")
        print(f"{'='*80}")
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i}. {row['Model']}")
            print(f"   R² Score: {row['R2']:.4f}")
            print(f"   RMSE: {row['RMSE']:.2f}")
            print(f"   MAE: {row['MAE']:.2f}")
            print(f"   MAPE: {row['MAPE']:.2f}%")
            print()
        
        # Best model summary
        best_model = results_df.iloc[0]
        print(f"🥇 BEST PERFORMING MODEL: {best_model['Model']}")
        print(f"   Explains {best_model['R2']*100:.1f}% of variance in {target_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_dataset_summary(available_datasets):
    """Print summary of available datasets"""
    print("\n" + "="*80)
    print("📋 DATASET SUMMARY")
    print("="*80)
    
    dataset_info = {
        "2016-building-energy-benchmarking.csv": {
            "name": "Seattle 2016 Building Energy Benchmarking",
            "source": "Seattle Office of Sustainability & Environment",
            "features": "Building details, energy consumption, emissions"
        },
        "2015-building-energy-benchmarking.csv": {
            "name": "Seattle 2015 Building Energy Benchmarking", 
            "source": "Seattle Office of Sustainability & Environment",
            "features": "Building details, energy consumption, emissions"
        },
        "energy_disclosure_2021_rows.csv": {
            "name": "NYC 2021 Energy Disclosure",
            "source": "New York City Department of Buildings",
            "features": "Building area, energy efficiency grades, ENERGY STAR scores"
        }
    }
    
    for dataset_path in available_datasets:
        filename = os.path.basename(dataset_path)
        if filename in dataset_info:
            info = dataset_info[filename]
            print(f"📊 {info['name']}")
            print(f"   Source: {info['source']}")
            print(f"   Features: {info['features']}")
            print(f"   File: {filename}")
            print()


def main():
    """
    Enhanced Main Pipeline for Multi-Dataset Processing
    """
    print_header("ENHANCED BUILDING ENERGY PREDICTION PIPELINE")
    print("🏢 Multi-Dataset Urban Building Energy Consumption Prediction")
    print("🤖 Advanced Machine Learning with Combined City Data")
    print("📊 Datasets: Seattle 2015/2016 + NYC 2021")
    
    # Record start time
    pipeline_start_time = time.time()
    
    # Create directories
    print("\n📁 Setting up project structure...")
    create_directories()
    
    # Check available datasets
    print("\n" + "="*80)
    available_datasets = check_datasets()
    
    if not available_datasets:
        print("\n❌ PIPELINE TERMINATED: No datasets found")
        print("\nPlease ensure at least one of these files exists in the 'data' directory:")
        print("  • data/2016-building-energy-benchmarking.csv")
        print("  • data/2015-building-energy-benchmarking.csv")
        print("  • data/energy_disclosure_2021_rows.csv")
        return
    
    # Print dataset summary
    print_dataset_summary(available_datasets)
    
    # Step 1: Enhanced Data Preprocessing
    print("\n" + "="*80)
    preprocessing_success = run_data_preprocessing(available_datasets)
    
    if not preprocessing_success:
        print("❌ PIPELINE TERMINATED: Preprocessing failed")
        return
    
    # Step 2: Enhanced Model Training
    print("\n" + "="*80)
    training_success = run_model_training()
    
    if not training_success:
        print("⚠️ Continuing to evaluation with available models...")
    
    # Step 3: Comprehensive Model Evaluation
    print("\n" + "="*80)
    evaluation_success = run_model_evaluation()
    
    # Final comprehensive summary
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print_header("ENHANCED PIPELINE EXECUTION SUMMARY")
    
    if preprocessing_success and training_success and evaluation_success:
        print("🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
        status = "✅ COMPLETE SUCCESS"
    elif preprocessing_success and evaluation_success:
        print("⚠️ PIPELINE COMPLETED WITH WARNINGS")
        status = "⚠️ PARTIAL SUCCESS"  
    else:
        print("❌ PIPELINE EXECUTION FAILED")
        status = "❌ FAILED"
    
    print(f"\n📊 Final Execution Summary:")
    print(f"   🎯 Status: {status}")
    print(f"   ⏱️  Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   📋 Datasets processed: {len(available_datasets)}")
    print(f"   🔄 Data preprocessing: {'✅' if preprocessing_success else '❌'}")
    print(f"   🤖 Model training: {'✅' if training_success else '❌'}")
    print(f"   📈 Model evaluation: {'✅' if evaluation_success else '❌'}")
    
    print(f"\n📁 Generated Output Files:")
    print(f"   📊 Preprocessed Data: outputs/X_train.csv, y_train.csv, X_test.csv, y_test.csv")
    print(f"   🤖 Trained Models: outputs/model_*.pkl")
    print(f"   🔮 Predictions: outputs/predictions_*.csv")
    print(f"   📈 Evaluation Results: outputs/model_evaluation_results.csv")
    print(f"   🎨 Visualizations: outputs/charts/*.png")
    print(f"   ℹ️  Metadata: outputs/feature_names.txt, target_info.txt")
    
    if os.path.exists("outputs/target_info.txt"):
        print(f"\n🎯 Model Performance Target:")
        with open("outputs/target_info.txt", "r") as f:
            lines = f.readlines()
            for line in lines[:4]:  # Show first 4 lines
                print(f"   {line.strip()}")
    
    print_header("ENHANCED PIPELINE COMPLETED", char="=")
    print("🏢 Ready for Building Energy Consumption Predictions!")


if __name__ == "__main__":
    main()