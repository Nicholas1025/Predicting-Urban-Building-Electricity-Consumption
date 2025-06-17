"""
Main Machine Learning Pipeline for Building Energy Consumption Prediction
Orchestrates data preprocessing, model training, and evaluation
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
from preprocessing.clean_data import preprocess_data
from evaluation.evaluate_models import (
    load_true_values, load_model_predictions, evaluate_all_models,
    plot_predicted_vs_actual, plot_model_comparison, plot_residuals_analysis,
    save_results_summary
)
from models.train_xgboost import main as train_xgb_main


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
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")


def train_xgboost_model():
    """
    Train XGBoost model
    """
    print("Training XGBoost model...")
    
    try:
        # Try to import from existing script
        from models.train_xgboost import main as train_xgb_main

        train_xgb_main()
    except ImportError:
        # Fallback: Create XGBoost model inline
        print("train_xgb.py not found. Training XGBoost model inline...")
        
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
            n_estimators=200,
            max_depth=6,
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
        
        print(f"  XGBoost Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Save model and predictions
        joblib.dump(model, "outputs/model_xgb.pkl")
        pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_xgb.csv", index=False)
        
        print("  ✓ XGBoost model saved")


# def train_random_forest_model():
#     """
#     Train Random Forest model
#     """
#     print("Training Random Forest model...")
    
#     try:
#         # Try to import from existing script
#         from train_rf import main as train_rf_main
#         train_rf_main()
#     except ImportError:
#         # Fallback: Create Random Forest model inline
#         print("train_rf.py not found. Training Random Forest model inline...")
        
#         from sklearn.ensemble import RandomForestRegressor
#         from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#         import joblib
        
#         # Load preprocessed data
#         X_train = pd.read_csv("outputs/X_train.csv")
#         X_test = pd.read_csv("outputs/X_test.csv")
#         y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
#         y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
#         # Create and train model
#         model = RandomForestRegressor(
#             n_estimators=100,
#             max_depth=15,
#             min_samples_split=5,
#             min_samples_leaf=2,
#             random_state=42,
#             n_jobs=-1
#         )
        
#         print("  Training Random Forest...")
#         model.fit(X_train, y_train)
        
#         # Make predictions
#         y_pred = model.predict(X_test)
        
#         # Calculate metrics
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         print(f"  Random Forest Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
#         # Save model and predictions
#         joblib.dump(model, "outputs/model_rf.pkl")
#         pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_rf.csv", index=False)
        
#         print("  ✓ Random Forest model saved")


# def train_svr_model():
#     """
#     Train Support Vector Regression model
#     """
#     print("Training SVR model...")
    
#     try:
#         # Try to import from existing script
#         from train_svr import main as train_svr_main
#         train_svr_main()
#     except ImportError:
#         # Fallback: Create SVR model inline
#         print("train_svr.py not found. Training SVR model inline...")
        
#         from sklearn.svm import SVR
#         from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#         import joblib
        
#         # Load preprocessed data
#         X_train = pd.read_csv("outputs/X_train.csv")
#         X_test = pd.read_csv("outputs/X_test.csv")
#         y_train = pd.read_csv("outputs/y_train.csv").iloc[:, 0]
#         y_test = pd.read_csv("outputs/y_test.csv").iloc[:, 0]
        
#         # For large datasets, we'll use a subset for SVR training due to computational constraints
#         if len(X_train) > 5000:
#             print("  Using subset of data for SVR training (computational efficiency)")
#             subset_indices = np.random.choice(len(X_train), 5000, replace=False)
#             X_train_subset = X_train.iloc[subset_indices]
#             y_train_subset = y_train.iloc[subset_indices]
#         else:
#             X_train_subset = X_train
#             y_train_subset = y_train
        
#         # Create and train model
#         model = SVR(
#             kernel='rbf',
#             C=100,
#             gamma='scale',
#             epsilon=0.1
#         )
        
#         print("  Training SVR...")
#         model.fit(X_train_subset, y_train_subset)
        
#         # Make predictions
#         y_pred = model.predict(X_test)
        
#         # Calculate metrics
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         print(f"  SVR Results: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
#         # Save model and predictions
#         joblib.dump(model, "outputs/model_svr.pkl")
#         pd.DataFrame({'predictions': y_pred}).to_csv("outputs/predictions_svr.csv", index=False)
        
#         print("  ✓ SVR model saved")


def run_data_preprocessing():
    """
    Step 1: Data Preprocessing
    """
    print_step(1, "DATA PREPROCESSING")
    
    # File path
    data_file = "data/2016-building-energy-benchmarking.csv"
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found at {data_file}")
        print("Please ensure the dataset is available at the specified path.")
        return False
    
    print(f"📁 Loading data from: {data_file}")
    
    try:
        # Run preprocessing
        result = preprocess_data(data_file)
        
        if result is not None:
            X_train, X_test, y_train, y_test, scaler, feature_names = result
            
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
            
            # Save scaler
            import joblib
            joblib.dump(scaler, "outputs/scaler.pkl")
            
            print("✅ Data preprocessing completed successfully!")
            print(f"   Training samples: {X_train.shape[0]}")
            print(f"   Test samples: {X_test.shape[0]}")
            print(f"   Features: {X_train.shape[1]}")
            
            return True
        else:
            print("❌ Data preprocessing failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False


def run_model_training():
    """
    Step 2: Model Training
    """
    print_step(2, "MODEL TRAINING")
    
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
        print("\n🚀 Training Model 1/3: XGBoost")
        start_time = time.time()
        train_xgboost_model()
        end_time = time.time()
        models_trained.append(f"XGBoost ({end_time - start_time:.1f}s)")
        print(f"   Training time: {end_time - start_time:.1f} seconds")
    except Exception as e:
        print(f"❌ XGBoost training failed: {e}")
    
    # # Train Random Forest
    # try:
    #     print("\n🌲 Training Model 2/3: Random Forest")
    #     start_time = time.time()
    #     train_random_forest_model()
    #     end_time = time.time()
    #     models_trained.append(f"Random Forest ({end_time - start_time:.1f}s)")
    #     print(f"   Training time: {end_time - start_time:.1f} seconds")
    # except Exception as e:
    #     print(f"❌ Random Forest training failed: {e}")
    
    # # Train SVR
    # try:
    #     print("\n⚡ Training Model 3/3: Support Vector Regression")
    #     start_time = time.time()
    #     train_svr_model()
    #     end_time = time.time()
    #     models_trained.append(f"SVR ({end_time - start_time:.1f}s)")
    #     print(f"   Training time: {end_time - start_time:.1f} seconds")
    # except Exception as e:
    #     print(f"❌ SVR training failed: {e}")
    
    if models_trained:
        print(f"\n✅ Model training completed!")
        print(f"   Models trained: {len(models_trained)}")
        for model in models_trained:
            print(f"   • {model}")
        return True
    else:
        print("❌ No models were trained successfully!")
        return False


def run_model_evaluation():
    """
    Step 3: Model Evaluation
    """
    print_step(3, "MODEL EVALUATION")
    
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
        
        print(f"✅ Found predictions for {len(predictions_dict)} models: {list(predictions_dict.keys())}")
        
        # Evaluate all models
        print("\n📈 Evaluating model performance...")
        results_df = evaluate_all_models(y_true, predictions_dict)
        
        # Create visualizations
        print("\n🎨 Creating visualizations...")
        plot_predicted_vs_actual(y_true, predictions_dict)
        plot_model_comparison(results_df)
        plot_residuals_analysis(y_true, predictions_dict)
        
        # Save results
        print("\n💾 Saving evaluation results...")
        save_results_summary(results_df)
        
        print("✅ Model evaluation completed successfully!")
        
        # Print final summary
        best_model = results_df.iloc[0]
        print(f"\n🏆 Best performing model: {best_model['Model']}")
        print(f"   R² Score: {best_model['R2']:.4f}")
        print(f"   RMSE: {best_model['RMSE']:.2f}")
        print(f"   MAE: {best_model['MAE']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False


def main():
    """
    Main pipeline execution
    """
    print_header("BUILDING ENERGY CONSUMPTION PREDICTION PIPELINE")
    print("🏢 Predicting urban building energy consumption using machine learning")
    print("📅 Dataset: Seattle Building Energy Benchmarking 2016")
    print("🤖 Models: XGBoost, Random Forest, Support Vector Regression")
    
    # Record start time
    pipeline_start_time = time.time()
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Step 1: Data Preprocessing
    print("\n" + "="*80)
    preprocessing_success = run_data_preprocessing()
    
    if not preprocessing_success:
        print("❌ Pipeline terminated due to preprocessing failure")
        return
    
    # Step 2: Model Training
    print("\n" + "="*80)
    training_success = run_model_training()
    
    if not training_success:
        print("⚠️ Continuing to evaluation with available models...")
    
    # Step 3: Model Evaluation
    print("\n" + "="*80)
    evaluation_success = run_model_evaluation()
    
    # Final summary
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    print_header("PIPELINE EXECUTION SUMMARY")
    
    if preprocessing_success and training_success and evaluation_success:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        status = "✅ SUCCESS"
    elif preprocessing_success and evaluation_success:
        print("⚠️ PIPELINE COMPLETED WITH WARNINGS")
        status = "⚠️ PARTIAL SUCCESS"
    else:
        print("❌ PIPELINE FAILED")
        status = "❌ FAILED"
    
    print(f"\n📊 Execution Summary:")
    print(f"   Status: {status}")
    print(f"   Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Data preprocessing: {'✅' if preprocessing_success else '❌'}")
    print(f"   Model training: {'✅' if training_success else '❌'}")
    print(f"   Model evaluation: {'✅' if evaluation_success else '❌'}")
    
    print(f"\n📁 Output files generated:")
    print(f"   Data: outputs/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print(f"   Models: outputs/model_*.pkl")
    print(f"   Predictions: outputs/predictions_*.csv")
    print(f"   Results: outputs/model_evaluation_results.csv")
    print(f"   Charts: outputs/charts/*.png")
    
    print_header("PIPELINE EXECUTION COMPLETED", char="=")


if __name__ == "__main__":
    main()