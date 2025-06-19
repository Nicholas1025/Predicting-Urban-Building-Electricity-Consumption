"""
Complete Project Runner for Building Energy Prediction System
Automated setup, execution, and validation of the entire ML pipeline
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def print_banner(title, char="=", width=80):
    """Print formatted banner"""
    print("\n" + char * width)
    print(f"{title.upper().center(width)}")
    print(char * width)


def print_section(title, char="-", width=60):
    """Print formatted section"""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  This project requires Python 3.8 or higher")
        print("Please upgrade Python and try again.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True


def check_required_packages():
    print("\n📦 Checking required packages...")
    
    package_mapping = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',   
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib',
        'flask': 'flask'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_mapping.items():
        try:
            __import__(import_name)  
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All required packages installed")
        return True


def check_data_files():
    """Check if required data files exist"""
    print("\n📁 Checking data files...")
    
    data_files = [
        "data/2015-building-energy-benchmarking.csv",
        "data/2016-building-energy-benchmarking.csv", 
        "data/energy_disclosure_2021_rows.csv"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            existing_files.append(file_path)
            
            # Check file size
            try:
                df = pd.read_csv(file_path, nrows=1)
                print(f"   📊 Columns: {len(df.columns)}")
            except Exception as e:
                print(f"   ⚠️  File may be corrupted: {e}")
        else:
            print(f"❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if not existing_files:
        print("\n❌ No data files found!")
        print("Please ensure at least one dataset is available in the 'data/' directory")
        return False
    elif missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files, but {len(existing_files)} available")
        print("Project will run with available datasets")
        return True
    else:
        print("✅ All data files found")
        return True


def create_directories():
    """Create necessary output directories"""
    print("\n📁 Creating output directories...")
    
    directories = [
        "outputs",
        "outputs/charts", 
        "outputs/tables",
        "outputs/models",
        "static"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}/")


def run_main_pipeline():
    """Execute the main ML pipeline"""
    print_section("RUNNING MAIN ML PIPELINE")
    
    try:
        print("🚀 Starting main.py pipeline...")
        start_time = time.time()
        
        # Import and run main pipeline
        from main import main as run_main
        result = run_main()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if result and result.get('overall_success', False):
            print(f"✅ Main pipeline completed successfully!")
            print(f"⏱️  Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
            return result
        else:
            print(f"⚠️  Main pipeline completed with some limitations")
            print(f"⏱️  Total execution time: {execution_time:.1f} seconds")
            return result or {'overall_success': False}
            
    except Exception as e:
        print(f"❌ Error running main pipeline: {e}")
        print("Please check the error above and ensure all dependencies are installed")
        return {'overall_success': False, 'error': str(e)}


def check_output_files():
    """Check what output files were generated"""
    print_section("CHECKING OUTPUT FILES")
    
    file_categories = {
        "📊 Data Files": [
            "outputs/X_train.csv",
            "outputs/X_test.csv", 
            "outputs/y_train.csv",
            "outputs/y_test.csv",
            "outputs/unified_features.csv",
            "outputs/unified_labels.csv"
        ],
        "🤖 Model Files": [
            "outputs/model_xgb.pkl",
            "outputs/model_rf.pkl",
            "outputs/model_svr.pkl",
            "outputs/models/model_xgboost_classifier.pkl",
            "outputs/models/model_random_forest_classifier.pkl",
            "outputs/models/model_svm_classifier.pkl"
        ],
        "📈 Prediction Files": [
            "outputs/predictions_xgb.csv",
            "outputs/predictions_rf.csv",
            "outputs/predictions_svr.csv",
            "outputs/predictions_xgboost_classification.csv",
            "outputs/predictions_random_forest_classification.csv",
            "outputs/predictions_svm_classification.csv"
        ],
        "📊 Chart Files": [
            "outputs/charts/predicted_vs_actual_all_models.png",
            "outputs/charts/model_comparison_metrics.png",
            "outputs/charts/residuals_analysis.png",
            "outputs/charts/feature_importance_xgb.png",
            "outputs/charts/feature_importance_rf.png",
            "outputs/charts/confusion_matrices_classification.png",
            "outputs/charts/classification_metrics_comparison.png"
        ],
        "📋 Table Files": [
            "outputs/model_evaluation_results.csv",
            "outputs/tables/classification_performance.csv",
            "outputs/tables/cross_year_summary.csv"
        ]
    }
    
    total_files = 0
    existing_files = 0
    
    for category, files in file_categories.items():
        print(f"\n{category}:")
        category_existing = 0
        
        for file_path in files:
            total_files += 1
            if os.path.exists(file_path):
                print(f"  ✅ {os.path.basename(file_path)}")
                existing_files += 1
                category_existing += 1
            else:
                print(f"  ❌ {os.path.basename(file_path)}")
        
        print(f"  📊 {category_existing}/{len(files)} files in this category")
    
    print(f"\n📊 OVERALL: {existing_files}/{total_files} files generated ({existing_files/total_files*100:.1f}%)")
    
    return existing_files, total_files


def validate_model_performance():
    """Validate model performance from results"""
    print_section("VALIDATING MODEL PERFORMANCE")
    
    try:
        # Check regression results
        regression_file = "outputs/model_evaluation_results.csv"
        if os.path.exists(regression_file):
            df_reg = pd.read_csv(regression_file)
            print("✅ Regression Results Found:")
            for _, row in df_reg.iterrows():
                print(f"  🤖 {row['Model']}: R² = {row['R2']:.4f}, RMSE = {row['RMSE']:.2f}")
        else:
            print("⚠️  No regression results found")
        
        # Check classification results  
        classification_file = "outputs/tables/classification_performance.csv"
        if os.path.exists(classification_file):
            df_class = pd.read_csv(classification_file)
            print("\n✅ Classification Results Found:")
            for _, row in df_class.iterrows():
                print(f"  🎯 {row['Model']}: Accuracy = {row['Accuracy']:.4f}, F1 = {row['F1-Score']:.4f}")
        else:
            print("\n⚠️  No classification results found")
            
    except Exception as e:
        print(f"❌ Error validating performance: {e}")


def start_web_dashboard():
    """Start the Flask web dashboard"""
    print_section("STARTING WEB DASHBOARD")
    
    try:
        print("🌐 Starting Flask web dashboard...")
        print("📍 Dashboard will be available at: http://localhost:5000")
        print("📊 Analytics page: http://localhost:5000/dashboard")
        print("\n🔧 Starting server...")
        print("💡 Press Ctrl+C to stop the server")
        
        # Import and run Flask app
        from app import app
        app.run(debug=True, host='localhost', port=5000)
        
    except ImportError:
        print("❌ Error: Flask app not found (app.py)")
        print("Please ensure app.py is in the current directory")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")


def generate_project_summary(pipeline_result):
    """Generate a comprehensive project summary"""
    print_section("PROJECT EXECUTION SUMMARY")
    
    existing_files, total_files = check_output_files()
    
    # Create summary
    summary = {
        'execution_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_success': pipeline_result.get('overall_success', False),
        'files_generated': f"{existing_files}/{total_files}",
        'completion_percentage': f"{existing_files/total_files*100:.1f}%"
    }
    
    # Save summary
    try:
        with open("outputs/project_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("✅ Project summary saved to: outputs/project_summary.json")
    except Exception as e:
        print(f"⚠️  Could not save summary: {e}")
    
    # Print summary
    print(f"\n📊 EXECUTION SUMMARY:")
    print(f"   🕐 Completed at: {summary['execution_time']}")
    print(f"   ✅ Pipeline success: {summary['pipeline_success']}")
    print(f"   📁 Files generated: {summary['files_generated']}")
    print(f"   📈 Completion: {summary['completion_percentage']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if summary['pipeline_success']:
        print("   🎉 Excellent! All core components completed successfully")
        print("   📊 Your model evaluation metrics are ready for reporting")
        print("   📈 Check the dashboard for detailed visualizations")
        print("   📋 Professional tables are available in outputs/tables/")
        
        if existing_files >= total_files * 0.9:
            print("   🏆 Outstanding! Nearly all expected files generated")
        elif existing_files >= total_files * 0.7:
            print("   ✅ Good! Most expected files generated")
        else:
            print("   ⚠️  Some optional files missing, but core results available")
    else:
        print("   ⚠️  Pipeline had some issues - check error messages above")
        print("   🔧 Try running individual components to identify issues")
        print("   📚 Ensure all data files are available and properly formatted")
    
    return summary


def interactive_menu():
    """Interactive menu for running different components"""
    while True:
        print_banner("BUILDING ENERGY PREDICTION - INTERACTIVE MENU")
        print("Choose an option:")
        print("1. 🔍 System Check (dependencies, data files)")
        print("2. 🚀 Run Complete Pipeline")
        print("3. 📊 Check Output Files")
        print("4. 🌐 Start Web Dashboard")
        print("5. 📋 View Project Summary")
        print("6. ❌ Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == "1":
                system_check()
            elif choice == "2":
                run_complete_pipeline()
            elif choice == "3":
                check_output_files()
            elif choice == "4":
                start_web_dashboard()
                break
            elif choice == "5":
                view_project_summary()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def system_check():
    """Run complete system check"""
    print_banner("SYSTEM CHECK")
    
    python_ok = check_python_version()
    packages_ok = check_required_packages()
    data_ok = check_data_files()
    
    create_directories()
    
    print(f"\n📊 SYSTEM CHECK SUMMARY:")
    print(f"   🐍 Python: {'✅' if python_ok else '❌'}")
    print(f"   📦 Packages: {'✅' if packages_ok else '❌'}")
    print(f"   📁 Data Files: {'✅' if data_ok else '❌'}")
    
    if python_ok and packages_ok and data_ok:
        print("🎉 System is ready! You can run the complete pipeline.")
    else:
        print("⚠️  Please resolve the issues above before running the pipeline.")


def run_complete_pipeline():
    """Run the complete ML pipeline"""
    print_banner("COMPLETE PIPELINE EXECUTION")
    
    # System check first
    print("Performing quick system check...")
    if not (check_python_version() and check_required_packages() and check_data_files()):
        print("❌ System check failed. Please resolve issues first.")
        return
    
    create_directories()
    
    # Run main pipeline
    pipeline_result = run_main_pipeline()
    
    # Validate results
    validate_model_performance()
    
    # Generate summary
    summary = generate_project_summary(pipeline_result)
    
    # Ask if user wants to start dashboard
    try:
        start_dashboard = input("\n🌐 Start web dashboard now? (y/n): ").lower().strip()
        if start_dashboard == 'y':
            start_web_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Pipeline completed. You can start the dashboard later with option 4.")


def view_project_summary():
    """View existing project summary"""
    summary_file = "outputs/project_summary.json"
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                summary = json.load(f)
            
            print_section("PROJECT SUMMARY")
            print(f"Execution Time: {summary.get('execution_time', 'Unknown')}")
            print(f"Pipeline Success: {summary.get('pipeline_success', 'Unknown')}")
            print(f"Files Generated: {summary.get('files_generated', 'Unknown')}")
            print(f"Completion: {summary.get('completion_percentage', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error reading summary: {e}")
    else:
        print("❌ No project summary found. Run the pipeline first.")


def main():
    """Main entry point"""
    print_banner("BUILDING ENERGY PREDICTION SYSTEM")
    print("🏢 Advanced ML Pipeline for Building Energy Consumption Analysis")
    print("📊 Multi-Model Comparison with Professional Reporting")
    print("🎯 XGBoost | Random Forest | SVR | Classification | Cross-Year Analysis")
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--auto', '-a']:
            print("🤖 Running in automatic mode...")
            run_complete_pipeline()
        elif arg in ['--check', '-c']:
            print("🔍 Running system check only...")
            system_check()
        elif arg in ['--dashboard', '-d']:
            print("🌐 Starting dashboard only...")
            start_web_dashboard()
        elif arg in ['--help', '-h']:
            print("\nUsage:")
            print("  python run_project.py           # Interactive menu")
            print("  python run_project.py --auto    # Run complete pipeline")
            print("  python run_project.py --check   # System check only")
            print("  python run_project.py --dashboard # Start dashboard only")
            print("  python run_project.py --help    # Show this help")
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Use --help for usage information")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error above and try again.")