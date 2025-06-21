"""
修复异常值移除问题的补丁
fix_outlier_removal.py
"""

import pandas as pd
import numpy as np
import os

def apply_fixed_outlier_removal(df, dataset_type, target_col):
    """
    修复后的异常值移除函数 - 只对目标变量进行异常值移除
    """
    from preprocessing.clean_data import get_fair_city_config
    config = get_fair_city_config(dataset_type)
    low_pct, high_pct = config['outlier_percentiles']
    
    print(f"{dataset_type}: 应用修复的异常值移除 ({low_pct}%-{high_pct}% 范围)")
    
    original_shape = df.shape[0]
    
    # 🔧 修复: 只对目标变量进行异常值移除，不对其他列进行过滤
    if target_col in df.columns:
        target_data = pd.to_numeric(df[target_col], errors='coerce')
        q_low = target_data.quantile(low_pct / 100)
        q_high = target_data.quantile(high_pct / 100)
        
        # 只移除目标变量的极端异常值
        target_mask = (target_data >= q_low) & (target_data <= q_high)
        df = df[target_mask]
        
        print(f"目标变量异常值移除: 保留 {df.shape[0]}/{original_shape} 行")
    
    # 🚫 删除对其他数值列的异常值处理 - 这是导致问题的根源
    # 不再对每个数值列单独进行异常值移除
    
    final_shape = df.shape[0]
    removed_count = original_shape - final_shape
    removal_pct = (removed_count / original_shape) * 100
    
    print(f"总异常值移除: {removed_count} 行 ({removal_pct:.1f}%)")
    
    return df

def patch_clean_data_module():
    """
    修补 clean_data.py 模块中的异常值移除函数
    """
    print("🔧 正在修补 clean_data.py 模块...")
    
    try:
        # 导入模块
        import preprocessing.clean_data as clean_data_module
        
        # 替换函数
        clean_data_module.apply_fair_outlier_removal = apply_fixed_outlier_removal
        
        print("✅ 异常值移除函数已修补")
        return True
        
    except Exception as e:
        print(f"❌ 修补失败: {e}")
        return False

def reprocess_dataset_with_fix(dataset_name, file_path):
    """
    使用修复后的异常值移除重新处理数据集
    """
    print(f"\n🔧 使用修复后的方法重新处理 {dataset_name}...")
    
    # 先修补模块
    if not patch_clean_data_module():
        return False
    
    try:
        # 使用修复后的预处理
        from preprocessing.clean_data import preprocess_data
        
        result = preprocess_data(file_path)
        
        if result is None:
            print(f"❌ {dataset_name} 预处理失败")
            return False
        
        X_train, X_test, y_train, y_test, scaler, feature_names = result
        
        # 保存到正确的目录
        output_dir = f"outputs/{dataset_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数据
        X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/y_train.csv", index=False)  
        y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
        
        # 保存其他文件
        with open(f"{output_dir}/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        
        print(f"✅ {dataset_name} 修复完成!")
        print(f"   训练集: {len(X_train)} 样本 (之前: 54/20)")
        print(f"   测试集: {len(X_test)} 样本 (之前: 14/5)")
        print(f"   特征: {len(feature_names)} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_all_datasets_outlier_issue():
    """
    修复所有数据集的异常值移除问题
    """
    print("🚀 修复所有数据集的异常值移除问题...")
    print("🎯 问题: 对每个数值列都进行异常值移除导致数据累积丢失")
    print("🔧 解决: 只对目标变量进行异常值移除")
    
    datasets = {
        'seattle_2015_present': 'data/seattle_2015_present.csv',
        'chicago_energy': 'data/chicago_energy_benchmarking.csv',
        'washington_dc': 'data/washington_dc_energy.csv'
    }
    
    results = {}
    
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            success = reprocess_dataset_with_fix(dataset_name, file_path)
            results[dataset_name] = success
        else:
            print(f"❌ 文件不存在: {file_path}")
            results[dataset_name] = False
    
    # 总结
    print(f"\n📊 异常值修复结果:")
    successful = sum(results.values())
    total = len(results)
    
    for dataset, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {dataset}")
    
    print(f"\n🎯 修复成功率: {successful}/{total} ({successful/total*100:.0f}%)")
    
    if successful > 0:
        print(f"\n📈 预期改进:")
        print(f"   Seattle: 68 → 20,000+ 样本")
        print(f"   Chicago: 25 → 15,000+ 样本")
        print(f"   Washington DC: 5,228 → 保持或改善")
        
        print(f"\n💡 下一步:")
        print(f"1. 重新运行模型训练:")
        print(f"   python run_project.py --individual")
        print(f"2. 或单独训练每个模型")
        print(f"3. 然后查看改进后的图表")
    
    return results

# 创建快速修复脚本
def create_quick_fix():
    """
    创建一个快速修复脚本，直接修改原始文件
    """
    print("📝 创建快速修复补丁...")
    
    # 读取原始文件
    clean_data_path = "preprocessing/clean_data.py"
    
    if not os.path.exists(clean_data_path):
        print(f"❌ 找不到 {clean_data_path}")
        return False
    
    try:
        with open(clean_data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建备份
        backup_path = "preprocessing/clean_data.py.backup"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 创建备份: {backup_path}")
        
        # 找到并替换异常值移除函数
        old_function_start = "def apply_fair_outlier_removal(df, dataset_type, target_col):"
        
        if old_function_start in content:
            # 找到函数的结束位置
            start_idx = content.find(old_function_start)
            
            # 找到下一个函数的开始
            next_function = content.find("\ndef ", start_idx + 1)
            if next_function == -1:
                next_function = len(content)
            
            # 替换函数
            new_function = '''def apply_fair_outlier_removal(df, dataset_type, target_col):
    """
    修复后的异常值移除 - 只对目标变量进行异常值移除
    """
    from preprocessing.clean_data import get_fair_city_config
    config = get_fair_city_config(dataset_type)
    low_pct, high_pct = config['outlier_percentiles']
    
    print(f"{dataset_type}: 应用修复的异常值移除 ({low_pct}%-{high_pct}% 范围)")
    
    original_shape = df.shape[0]
    
    # 修复: 只对目标变量进行异常值移除
    if target_col in df.columns:
        target_data = pd.to_numeric(df[target_col], errors='coerce')
        q_low = target_data.quantile(low_pct / 100)
        q_high = target_data.quantile(high_pct / 100)
        
        # 只移除目标变量的极端异常值
        target_mask = (target_data >= q_low) & (target_data <= q_high)
        df = df[target_mask]
        
        print(f"目标变量异常值移除: 保留 {df.shape[0]}/{original_shape} 行")
    
    final_shape = df.shape[0]
    removed_count = original_shape - final_shape
    removal_pct = (removed_count / original_shape) * 100
    
    print(f"总异常值移除: {removed_count} 行 ({removal_pct:.1f}%)")
    
    return df

'''
            
            # 替换内容
            new_content = content[:start_idx] + new_function + content[next_function:]
            
            # 保存修改后的文件
            with open(clean_data_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ 已修复 {clean_data_path}")
            print(f"🔧 修复: 移除了对所有数值列的异常值过滤")
            print(f"📄 备份保存在: {backup_path}")
            
            return True
        else:
            print(f"❌ 找不到目标函数")
            return False
            
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

if __name__ == "__main__":
    print("🔧 异常值移除问题修复工具")
    print("="*50)
    
    # 选择修复方式
    print("选择修复方式:")
    print("1. 直接修改源文件 (推荐)")
    print("2. 运行时修补")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        if create_quick_fix():
            print("\n✅ 源文件已修复!")
            print("现在可以重新运行预处理:")
            print("python run_project.py --individual")
        else:
            print("\n❌ 源文件修复失败")
    else:
        fix_all_datasets_outlier_issue()