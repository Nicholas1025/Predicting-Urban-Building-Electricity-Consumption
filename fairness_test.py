"""
Quick Fairness Test Script
Tests the new fair preprocessing on all three datasets to ensure equal treatment
"""

import sys
import os
import pandas as pd
import time
sys.path.append('.')

def test_fair_preprocessing():
    """Test fair preprocessing on all datasets"""
    
    print("="*80)
    print("FAIRNESS TEST - EQUAL TREATMENT PREPROCESSING")
    print("="*80)
    
    datasets = {
        'Seattle 2015-Present': 'data/seattle_2015_present.csv',
        'Chicago Energy': 'data/chicago_energy_benchmarking.csv', 
        'Washington DC': 'data/washington_dc_energy.csv'
    }
    
    results = {}
    
    for name, file_path in datasets.items():
        print(f"\n{'-'*60}")
        print(f"TESTING: {name}")
        print(f"File: {file_path}")
        print('-'*60)
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            results[name] = {'status': 'missing', 'error': 'File not found'}
            continue
        
        try:
            # Test preprocessing
            start_time = time.time()
            
            from preprocessing.clean_data import preprocess_data
            result = preprocess_data(file_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result:
                X_train, X_test, y_train, y_test, scaler, feature_names = result
                
                results[name] = {
                    'status': 'success',
                    'samples_train': len(X_train),
                    'samples_test': len(X_test),
                    'total_samples': len(X_train) + len(X_test),
                    'features': len(feature_names),
                    'target_range': (float(y_train.min()), float(y_train.max())),
                    'processing_time': processing_time
                }
                
                print(f"✅ SUCCESS")
                print(f"   📊 Total samples: {results[name]['total_samples']}")
                print(f"   🎯 Train samples: {results[name]['samples_train']}")
                print(f"   🧪 Test samples: {results[name]['samples_test']}")
                print(f"   📋 Features: {results[name]['features']}")
                print(f"   📈 Target range: [{results[name]['target_range'][0]:.0f}, {results[name]['target_range'][1]:.0f}]")
                print(f"   ⏱️  Processing time: {processing_time:.2f}s")
                
            else:
                results[name] = {'status': 'failed', 'error': 'No result returned'}
                print(f"❌ FAILED: No result returned")
                
        except Exception as e:
            results[name] = {'status': 'error', 'error': str(e)}
            print(f"❌ ERROR: {e}")
    
    # Generate fairness report
    print(f"\n{'='*80}")
    print("FAIRNESS ASSESSMENT REPORT")
    print('='*80)
    
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] != 'success']
    
    print(f"📊 SUCCESS RATE: {len(successful)}/{len(datasets)} ({len(successful)/len(datasets)*100:.1f}%)")
    
    if successful:
        print(f"\n✅ SUCCESSFUL DATASETS:")
        
        # Compare sample retention rates
        print(f"\n📈 SAMPLE RETENTION COMPARISON:")
        for name in successful:
            result = results[name]
            print(f"   {name}: {result['total_samples']} samples, {result['features']} features")
        
        # Compare processing times
        print(f"\n⏱️  PROCESSING TIME COMPARISON:")
        for name in successful:
            result = results[name]
            print(f"   {name}: {result['processing_time']:.2f}s")
        
        # Check for fairness metrics
        sample_counts = [results[name]['total_samples'] for name in successful]
        feature_counts = [results[name]['features'] for name in successful]
        
        print(f"\n🎯 FAIRNESS METRICS:")
        print(f"   Sample count range: {min(sample_counts)} - {max(sample_counts)}")
        print(f"   Feature count range: {min(feature_counts)} - {max(feature_counts)}")
        
        # Check if results are more balanced
        sample_ratio = max(sample_counts) / min(sample_counts) if min(sample_counts) > 0 else float('inf')
        feature_ratio = max(feature_counts) / min(feature_counts) if min(feature_counts) > 0 else float('inf')
        
        print(f"   Sample count ratio: {sample_ratio:.2f} (lower is more fair)")
        print(f"   Feature count ratio: {feature_ratio:.2f} (lower is more fair)")
        
        if sample_ratio < 5 and feature_ratio < 3:
            print("   ✅ GOOD: Reasonably balanced across datasets")
        else:
            print("   ⚠️  CONCERN: Significant imbalance detected")
    
    if failed:
        print(f"\n❌ FAILED DATASETS:")
        for name in failed:
            result = results[name]
            print(f"   {name}: {result.get('error', 'Unknown error')}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(successful) >= 2:
        print("   🎉 Good! Multiple datasets processed successfully")
        print("   📊 You can proceed with comparative analysis")
        print("   🏙️ Each city will have fair treatment in the analysis")
    elif len(successful) == 1:
        print("   ⚠️  Only one dataset successful - limited analysis possible")
        print("   🔧 Check data files for the failed datasets")
    else:
        print("   ❌ No datasets processed successfully")
        print("   📁 Please check that data files exist and are properly formatted")
    
    return results


def compare_with_old_results():
    """Compare new fair results with old results if available"""
    
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS RESULTS")
    print('='*80)
    
    # Check if old results exist
    old_logs = []
    if os.path.exists("logs"):
        for file in os.listdir("logs"):
            if file.startswith("analysis_") and file.endswith(".log"):
                old_logs.append(file)
    
    if old_logs:
        print(f"📄 Found {len(old_logs)} previous analysis logs")
        print("   You can compare the new fair results with previous runs")
        print("   Expected improvements:")
        print("   ✅ More balanced sample counts across cities")
        print("   ✅ Higher success rate for Chicago and Washington DC")
        print("   ✅ Better model performance for previously failing datasets")
    else:
        print("📄 No previous analysis logs found - this is a fresh start")
    
    print("\n🔄 After running the full analysis, you can compare:")
    print("   📊 Model R² scores across cities")
    print("   📈 Sample retention rates")  
    print("   ⏱️  Processing times")
    print("   🎯 Classification accuracy")


if __name__ == "__main__":
    print("🔬 Running fairness test for building energy prediction preprocessing...")
    
    # Run the test
    test_results = test_fair_preprocessing()
    
    # Compare with old results
    compare_with_old_results()
    
    print(f"\n🏁 Fairness test completed!")
    print("💡 If tests pass, run the full analysis with: python run_project.py --individual")