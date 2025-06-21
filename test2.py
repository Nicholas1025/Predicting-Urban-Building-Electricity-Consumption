"""
修复数据丢失问题
fix_data_loss.py - 诊断并修复预处理中的数据丢失
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def debug_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    带调试信息的预处理函数，找出数据丢失的地方
    """
    print(f"🔍 调试预处理: {file_path}")
    
    # 1. 加载数据
    df = pd.read_csv(file_path)
    print(f"1️⃣ 加载原始数据: {df.shape}")
    
    # 2. 识别数据集类型
    from preprocessing.clean_data import identify_dataset_type, preprocess_data_city_specific
    dataset_type = identify_dataset_type(df, file_path)
    print(f"2️⃣ 数据集类型: {dataset_type}")
    
    # 3. 城市特定预处理
    df_processed, target_col = preprocess_data_city_specific(df, dataset_type)
    print(f"3️⃣ 城市预处理后: {df_processed.shape}")
    print(f"    目标变量: {target_col}")
    
    # 4. 分离特征和目标
    y = df_processed[target_col].copy()
    X = df_processed.drop(columns=[target_col])
    print(f"4️⃣ 分离特征目标: X={X.shape}, y={y.shape}")
    
    # 5. 删除无关列
    from preprocessing.clean_data import get_fair_city_config
    config = get_fair_city_config(dataset_type)
    irrelevant_cols = config['irrelevant_cols']
    
    existing_irrelevant = [col for col in irrelevant_cols if col in X.columns]
    if existing_irrelevant:
        X = X.drop(columns=existing_irrelevant)
        print(f"5️⃣ 删除无关列后: {X.shape} (删除了 {len(existing_irrelevant)} 列)")
    
    # 6. 处理分类变量
    from sklearn.preprocessing import LabelEncoder
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print(f"6️⃣ 处理分类变量: 找到 {len(categorical_cols)} 个分类列")
    
    for col in categorical_cols:
        unique_count = X[col].nunique()
        if unique_count > 100:
            X = X.drop(columns=[col])
            print(f"    删除 {col} - 类别太多 ({unique_count})")
        elif unique_count <= 1:
            X = X.drop(columns=[col])
            print(f"    删除 {col} - 常数列")
        else:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"    编码 {col} - {unique_count} 个类别")
    
    print(f"6️⃣ 分类处理后: {X.shape}")
    
    # 7. 处理数值变量的缺失值
    numerical_cols = X.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        missing_before = X[numerical_cols].isnull().sum().sum()
        for col in numerical_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)
        missing_after = X[numerical_cols].isnull().sum().sum()
        print(f"7️⃣ 处理缺失值: {missing_before} → {missing_after}")
    
    # 8. 删除常数列
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"8️⃣ 删除常数列: {len(constant_cols)} 列，剩余 {X.shape}")
    
    # 9. 删除高相关性特征
    if len(X.columns) > 1:
        try:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_triangle.columns 
                                 if any(upper_triangle[column] > 0.95)]
            if high_corr_features:
                X = X.drop(columns=high_corr_features)
                print(f"9️⃣ 删除高相关特征: {len(high_corr_features)} 列，剩余 {X.shape}")
        except Exception as e:
            print(f"9️⃣ 相关性分析失败: {e}")
    
    print(f"🔟 最终特征数量: {X.shape[1]}")
    
    if X.shape[1] == 0:
        print("❌ 所有特征都被删除了!")
        return None
    
    # 🚨 这里可能是问题所在 - 检查数据对齐
    print(f"📊 检查数据对齐:")
    print(f"   X索引范围: {X.index.min()} - {X.index.max()}")
    print(f"   y索引范围: {y.index.min()} - {y.index.max()}")
    
    # 确保X和y索引对齐
    common_indices = X.index.intersection(y.index)
    if len(common_indices) != len(X):
        print(f"⚠️ 索引不对齐! 公共索引: {len(common_indices)}, X长度: {len(X)}")
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        print(f"📊 对齐后: X={X.shape}, y={y.shape}")
    
    # 10. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"🔄 数据分割:")
    print(f"   训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"   测试集: X={X_test.shape}, y={y_test.shape}")
    
    # 11. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 转换回DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"📏 标准化完成:")
    print(f"   X_train: {X_train_scaled.shape}")
    print(f"   X_test: {X_test_scaled.shape}")
    
    feature_names = X_train.columns.tolist()
    
    print(f"✅ 预处理完成!")
    print(f"   最终训练集: {X_train_scaled.shape[0]} 样本")
    print(f"   最终测试集: {X_test_scaled.shape[0]} 样本")
    print(f"   特征数量: {len(feature_names)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def fix_preprocessing_for_dataset(dataset_name, file_path):
    """
    修复特定数据集的预处理问题
    """
    print(f"\n🔧 修复 {dataset_name} 的预处理...")
    
    try:
        # 使用调试预处理
        result = debug_preprocess_data(file_path)
        
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
        
        # 保存特征名称
        with open(f"{output_dir}/feature_names.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # 保存scaler
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        
        print(f"✅ {dataset_name} 修复完成!")
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        print(f"   特征: {len(feature_names)} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_all_datasets():
    """
    修复所有数据集的数据丢失问题
    """
    print("🚀 开始修复所有数据集的数据丢失问题...")
    
    datasets = {
        'seattle_2015_present': 'data/seattle_2015_present.csv',
        'chicago_energy': 'data/chicago_energy_benchmarking.csv',
        'washington_dc': 'data/washington_dc_energy.csv'
    }
    
    results = {}
    
    for dataset_name, file_path in datasets.items():
        if os.path.exists(file_path):
            success = fix_preprocessing_for_dataset(dataset_name, file_path)
            results[dataset_name] = success
        else:
            print(f"❌ 文件不存在: {file_path}")
            results[dataset_name] = False
    
    # 总结
    print(f"\n📊 修复结果:")
    successful = sum(results.values())
    total = len(results)
    
    for dataset, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {dataset}")
    
    print(f"\n🎯 成功率: {successful}/{total} ({successful/total*100:.0f}%)")
    
    if successful > 0:
        print(f"\n💡 建议:")
        print(f"1. 重新运行模型训练:")
        for dataset, success in results.items():
            if success:
                print(f"   python models/train_xgboost_individual.py {dataset}")
                print(f"   python models/train_rf_individual.py {dataset}")
                print(f"   python models/train_svr_individual.py {dataset}")
        
        print(f"2. 重新运行评估:")
        for dataset, success in results.items():
            if success:
                print(f"   python evaluation/evaluate_models_individual.py {dataset}")
    
    return results

if __name__ == "__main__":
    fix_all_datasets()