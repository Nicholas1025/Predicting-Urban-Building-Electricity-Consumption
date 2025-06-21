"""
完整的数据检查脚本
check_saved_data.py - 检查预处理后保存的数据文件
"""

import pandas as pd
import numpy as np
import os
import glob

def check_dataset_files(dataset_name):
    """检查单个数据集的所有相关文件"""
    
    print(f"\n{'='*70}")
    print(f"📊 检查数据集: {dataset_name.upper()}")
    print('='*70)
    
    data_dir = f"outputs/{dataset_name}"
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据集目录不存在: {data_dir}")
        return False
    
    # 1. 检查基础数据文件
    print(f"\n📁 基础数据文件:")
    basic_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    
    file_sizes = {}
    for file_name in basic_files:
        file_path = f"{data_dir}/{file_name}"
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                file_sizes[file_name] = len(df)
                features = df.shape[1] if len(df.shape) > 1 else 1
                print(f"   ✅ {file_name}: {len(df)} 行, {features} 列")
            except Exception as e:
                print(f"   ❌ {file_name}: 读取错误 - {e}")
                file_sizes[file_name] = 0
        else:
            print(f"   ❌ {file_name}: 文件不存在")
            file_sizes[file_name] = 0
    
    # 2. 检查训练/测试比例
    if file_sizes['X_train.csv'] > 0 and file_sizes['X_test.csv'] > 0:
        total_samples = file_sizes['X_train.csv'] + file_sizes['X_test.csv']
        train_ratio = file_sizes['X_train.csv'] / total_samples
        test_ratio = file_sizes['X_test.csv'] / total_samples
        
        print(f"\n📊 数据分割比例:")
        print(f"   训练集: {file_sizes['X_train.csv']} 样本 ({train_ratio:.1%})")
        print(f"   测试集: {file_sizes['X_test.csv']} 样本 ({test_ratio:.1%})")
        print(f"   总计: {total_samples} 样本")
        
        # 检查比例是否合理
        if 0.15 <= test_ratio <= 0.25:
            print(f"   ✅ 分割比例正常")
        else:
            print(f"   ⚠️ 分割比例异常 (期望测试集占15-25%)")
    
    # 3. 检查目标变量
    print(f"\n🎯 目标变量检查:")
    if file_sizes['y_train.csv'] > 0:
        try:
            y_train = pd.read_csv(f"{data_dir}/y_train.csv").iloc[:, 0]
            print(f"   训练目标: {len(y_train)} 值")
            print(f"   范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
            print(f"   均值: {y_train.mean():.2f}")
            print(f"   标准差: {y_train.std():.2f}")
        except Exception as e:
            print(f"   ❌ 目标变量读取错误: {e}")
    
    if file_sizes['y_test.csv'] > 0:
        try:
            y_test = pd.read_csv(f"{data_dir}/y_test.csv").iloc[:, 0]
            print(f"   测试目标: {len(y_test)} 值")
            print(f"   范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
        except Exception as e:
            print(f"   ❌ 测试目标读取错误: {e}")
    
    # 4. 检查模型文件
    print(f"\n🤖 模型文件检查:")
    model_files = ['model_xgb.pkl', 'model_rf.pkl', 'model_svr.pkl']
    models_dir = f"{data_dir}/models"
    
    model_count = 0
    if os.path.exists(models_dir):
        for model_file in model_files:
            model_path = f"{models_dir}/{model_file}"
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024*1024)
                print(f"   ✅ {model_file}: {size_mb:.2f} MB")
                model_count += 1
            else:
                print(f"   ❌ {model_file}: 不存在")
    else:
        print(f"   ❌ 模型目录不存在: {models_dir}")
    
    # 5. 检查预测文件
    print(f"\n📈 预测文件检查:")
    prediction_files = ['predictions_xgb.csv', 'predictions_rf.csv', 'predictions_svr.csv']
    
    prediction_count = 0
    for pred_file in prediction_files:
        pred_path = f"{data_dir}/{pred_file}"
        if os.path.exists(pred_path):
            try:
                pred_df = pd.read_csv(pred_path)
                prediction_count += 1
                print(f"   ✅ {pred_file}: {len(pred_df)} 预测值")
                
                # 检查预测值是否合理
                predictions = pred_df.iloc[:, 0]
                if predictions.notna().sum() == len(predictions):
                    print(f"       范围: [{predictions.min():.2f}, {predictions.max():.2f}]")
                else:
                    nan_count = predictions.isna().sum()
                    print(f"       ⚠️ 包含 {nan_count} 个NaN值")
                    
            except Exception as e:
                print(f"   ❌ {pred_file}: 读取错误 - {e}")
        else:
            print(f"   ❌ {pred_file}: 不存在")
    
    # 6. 检查图表文件
    print(f"\n📊 图表文件检查:")
    charts_dir = f"{data_dir}/charts"
    chart_count = 0
    
    if os.path.exists(charts_dir):
        chart_files = glob.glob(f"{charts_dir}/*.png")
        chart_count = len(chart_files)
        print(f"   找到 {chart_count} 个图表文件:")
        
        important_charts = [
            'predicted_vs_actual_all_models.png',
            'model_comparison_metrics.png',
            'feature_importance_xgb.png',
            'feature_importance_rf.png'
        ]
        
        for chart_name in important_charts:
            chart_path = f"{charts_dir}/{chart_name}"
            if os.path.exists(chart_path):
                size_kb = os.path.getsize(chart_path) / 1024
                print(f"   ✅ {chart_name}: {size_kb:.1f} KB")
            else:
                print(f"   ❌ {chart_name}: 不存在")
    else:
        print(f"   ❌ 图表目录不存在: {charts_dir}")
    
    # 7. 检查评估结果
    print(f"\n📋 评估结果检查:")
    results_file = f"{data_dir}/model_evaluation_results.csv"
    
    if os.path.exists(results_file):
        try:
            results_df = pd.read_csv(results_file)
            print(f"   ✅ 评估结果: {len(results_df)} 个模型")
            
            for _, row in results_df.iterrows():
                model_name = row['Model']
                r2_score = row['R2']
                rmse = row['RMSE']
                print(f"       {model_name}: R² = {r2_score:.4f}, RMSE = {rmse:.2f}")
                
        except Exception as e:
            print(f"   ❌ 评估结果读取错误: {e}")
    else:
        print(f"   ❌ 评估结果文件不存在")
    
    # 8. 数据集健康状况总结
    print(f"\n💊 数据集健康状况:")
    health_score = 0
    max_score = 7
    
    # 基础数据完整性
    if all(file_sizes[f] > 0 for f in basic_files):
        health_score += 2
        print(f"   ✅ 基础数据完整 (+2分)")
    else:
        missing = [f for f in basic_files if file_sizes[f] == 0]
        print(f"   ❌ 基础数据缺失: {missing}")
    
    # 模型训练完成度
    if model_count >= 2:
        health_score += 1
        print(f"   ✅ 模型训练充分 (+1分)")
    else:
        print(f"   ⚠️ 模型训练不足 ({model_count}/3)")
    
    # 预测生成完成度
    if prediction_count >= 2:
        health_score += 1
        print(f"   ✅ 预测生成充分 (+1分)")
    else:
        print(f"   ⚠️ 预测生成不足 ({prediction_count}/3)")
    
    # 图表生成完成度
    if chart_count >= 3:
        health_score += 1
        print(f"   ✅ 图表生成充分 (+1分)")
    else:
        print(f"   ⚠️ 图表生成不足 ({chart_count} 个)")
    
    # 评估完成度
    if os.path.exists(results_file):
        health_score += 1
        print(f"   ✅ 评估完成 (+1分)")
    else:
        print(f"   ❌ 评估未完成")
    
    # 数据量合理性
    total_samples = file_sizes.get('X_train.csv', 0) + file_sizes.get('X_test.csv', 0)
    if total_samples > 1000:
        health_score += 1
        print(f"   ✅ 数据量充足 (+1分)")
    elif total_samples > 100:
        print(f"   ⚠️ 数据量一般 ({total_samples} 样本)")
    else:
        print(f"   ❌ 数据量不足 ({total_samples} 样本)")
    
    # 健康状况评级
    health_percentage = (health_score / max_score) * 100
    
    if health_percentage >= 85:
        status = "🟢 优秀"
    elif health_percentage >= 70:
        status = "🟡 良好"
    elif health_percentage >= 50:
        status = "🟠 一般"
    else:
        status = "🔴 需要修复"
    
    print(f"\n🏥 总体健康度: {health_score}/{max_score} ({health_percentage:.0f}%) - {status}")
    
    return health_score >= max_score * 0.5

def check_all_datasets():
    """检查所有数据集"""
    
    print("🔍 开始完整数据检查...")
    print("检查预处理后保存的所有数据文件")
    
    datasets = ['seattle_2015_present', 'chicago_energy', 'washington_dc']
    
    # 检查outputs目录
    if not os.path.exists('outputs'):
        print("❌ outputs目录不存在!")
        print("请先运行数据预处理和模型训练")
        return
    
    print(f"✅ 找到outputs目录")
    
    # 检查每个数据集
    healthy_datasets = 0
    total_datasets = 0
    
    for dataset in datasets:
        if os.path.exists(f"outputs/{dataset}"):
            is_healthy = check_dataset_files(dataset)
            if is_healthy:
                healthy_datasets += 1
            total_datasets += 1
        else:
            print(f"\n❌ 数据集目录不存在: outputs/{dataset}")
    
    # 总体总结
    print(f"\n{'='*70}")
    print(f"🎯 总体检查结果")
    print('='*70)
    
    print(f"📊 数据集状态:")
    print(f"   总数据集: {len(datasets)}")
    print(f"   已处理: {total_datasets}")
    print(f"   健康: {healthy_datasets}")
    print(f"   成功率: {healthy_datasets/len(datasets)*100:.1f}%")
    
    if healthy_datasets == len(datasets):
        print(f"🎉 所有数据集都健康!")
        print(f"📈 可以正常查看dashboard和结果")
    elif healthy_datasets > 0:
        print(f"⚠️ 部分数据集有问题")
        print(f"🔧 建议重新运行有问题的数据集")
    else:
        print(f"❌ 所有数据集都有问题")
        print(f"🔧 建议重新运行完整的分析流程")
    
    # 给出具体建议
    print(f"\n💡 建议:")
    print(f"1. 如果基础数据缺失 → 重新运行预处理")
    print(f"2. 如果模型缺失 → 重新运行模型训练")
    print(f"3. 如果图表缺失 → 重新运行评估脚本")
    print(f"4. 如果数据量太少 → 检查原始数据文件")

def check_specific_issue():
    """专门检查你遇到的图表数据点少的问题"""
    
    print(f"\n{'='*70}")
    print(f"🔍 专项检查: 图表数据点问题")
    print('='*70)
    
    datasets = ['seattle_2015_present', 'chicago_energy', 'washington_dc']
    
    for dataset in datasets:
        print(f"\n📊 {dataset.upper()} - 图表数据检查:")
        
        # 检查测试集大小
        test_file = f"outputs/{dataset}/y_test.csv"
        if os.path.exists(test_file):
            y_test = pd.read_csv(test_file).iloc[:, 0]
            print(f"   🎯 测试集大小: {len(y_test)} 个样本")
            
            # 检查预测文件
            pred_files = ['predictions_xgb.csv', 'predictions_rf.csv', 'predictions_svr.csv']
            
            for pred_file in pred_files:
                pred_path = f"outputs/{dataset}/{pred_file}"
                if os.path.exists(pred_path):
                    try:
                        pred_df = pd.read_csv(pred_path)
                        pred_values = pred_df.iloc[:, 0]
                        
                        print(f"   📈 {pred_file}: {len(pred_values)} 个预测值")
                        
                        # 检查数据匹配
                        if len(pred_values) == len(y_test):
                            print(f"       ✅ 数据长度匹配")
                        else:
                            print(f"       ❌ 数据长度不匹配! 预测:{len(pred_values)} vs 真实:{len(y_test)}")
                        
                        # 检查是否有异常值
                        valid_preds = pred_values.notna().sum()
                        if valid_preds == len(pred_values):
                            print(f"       ✅ 所有预测值有效")
                        else:
                            print(f"       ⚠️ {len(pred_values) - valid_preds} 个预测值无效")
                            
                    except Exception as e:
                        print(f"       ❌ 读取错误: {e}")
                else:
                    print(f"   ❌ {pred_file} 不存在")
        else:
            print(f"   ❌ 测试集文件不存在")

if __name__ == "__main__":
    check_all_datasets()
    check_specific_issue()
    
    print(f"\n🏁 完整检查完成!")
    print(f"📋 如果发现问题，请将结果发给我，我会提供具体的修复方案")