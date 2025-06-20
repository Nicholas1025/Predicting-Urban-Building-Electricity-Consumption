"""
Flask Web Dashboard for Building Energy Prediction
FIXED VERSION v2 - Fixed chart display issues
"""

from flask import Flask, render_template, send_from_directory, jsonify, abort, request
import os
import pandas as pd
import json

app = Flask(__name__)

# 定义数据集
DATASETS = ['seattle_2015', 'seattle_2016', 'nyc_2021']

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """仪表板页面"""
    return render_template('dashboard.html')

@app.route('/outputs/<dataset>/<path:filename>')
def serve_dataset_file(dataset, filename):
    """
    提供独立数据集文件服务 - FIXED VERSION
    处理路径如: /outputs/seattle_2015/charts/model_comparison.png
    """
    print(f"🔍 Requested: /outputs/{dataset}/{filename}")
    
    if dataset not in DATASETS:
        print(f"❌ Invalid dataset: {dataset}")
        abort(404)
    
    # 构建完整的文件路径
    base_dir = os.path.join('outputs', dataset)
    full_path = os.path.join(base_dir, filename)
    
    print(f"📁 Looking for file: {full_path}")
    print(f"📁 File exists: {os.path.exists(full_path)}")
    
    # 检查文件是否存在
    if not os.path.exists(full_path):
        print(f"❌ File not found: {full_path}")
        # 列出可用文件进行调试
        if os.path.exists(base_dir):
            print(f"📂 Available files in {base_dir}:")
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                    print(f"   📄 {rel_path}")
        abort(404)
    
    # 分离目录和文件名
    file_dir = os.path.dirname(filename)
    file_name = os.path.basename(filename)
    
    if file_dir:
        serve_dir = os.path.join(base_dir, file_dir)
    else:
        serve_dir = base_dir
        
    print(f"✅ Serving from: {serve_dir}/{file_name}")
    
    try:
        return send_from_directory(serve_dir, file_name)
    except Exception as e:
        print(f"❌ Error serving file: {e}")
        abort(500)

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    """
    提供通用输出文件服务 - BACKWARD COMPATIBILITY
    尝试从第一个可用的数据集中提供文件
    """
    print(f"🔍 Legacy route requested: /outputs/{filename}")
    
    # 尝试从每个数据集中找到文件
    for dataset in DATASETS:
        file_path = os.path.join('outputs', dataset, filename)
        print(f"🔍 Trying: {file_path}")
        
        if os.path.exists(file_path):
            print(f"✅ Found in {dataset}: {file_path}")
            # 重定向到正确的数据集路由
            return serve_dataset_file(dataset, filename)
    
    # 检查根outputs目录
    root_file_path = os.path.join('outputs', filename)
    if os.path.exists(root_file_path):
        print(f"✅ Found in root: {root_file_path}")
        return send_from_directory('outputs', filename)
    
    print(f"❌ File not found anywhere: {filename}")
    abort(404)

@app.route('/api/datasets')
def get_datasets():
    """API: 获取可用的数据集信息"""
    datasets_info = {}
    
    for dataset in DATASETS:
        dataset_dir = f"outputs/{dataset}"
        print(f"🔍 Checking dataset: {dataset} -> {dataset_dir}")
        
        if os.path.exists(dataset_dir):
            datasets_info[dataset] = {
                'available': True,
                'charts': [],
                'models': [],
                'results': {}
            }
            
            # 检查图表文件
            charts_dir = os.path.join(dataset_dir, 'charts')
            if os.path.exists(charts_dir):
                chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
                datasets_info[dataset]['charts'] = chart_files
                print(f"📊 Found {len(chart_files)} charts for {dataset}")
            else:
                print(f"📊 No charts directory for {dataset}")
            
            # 检查模型文件
            models_dir = os.path.join(dataset_dir, 'models')
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                datasets_info[dataset]['models'] = model_files
                print(f"🤖 Found {len(model_files)} models for {dataset}")
            
            # 检查结果文件
            results_file = os.path.join(dataset_dir, 'model_evaluation_results.csv')
            if os.path.exists(results_file):
                try:
                    df = pd.read_csv(results_file)
                    datasets_info[dataset]['results'] = df.to_dict('records')
                    print(f"📈 Found evaluation results for {dataset}")
                except Exception as e:
                    datasets_info[dataset]['results'] = {'error': str(e)}
                    print(f"❌ Error reading results for {dataset}: {e}")
        else:
            datasets_info[dataset] = {'available': False}
            print(f"❌ Dataset not available: {dataset}")
    
    return jsonify(datasets_info)

@app.route('/api/dataset/<dataset>/results')
def get_dataset_results(dataset):
    """API: 获取特定数据集的结果"""
    if dataset not in DATASETS:
        return jsonify({'error': 'Dataset not found'}), 404
    
    results = {}
    dataset_dir = f"outputs/{dataset}"
    
    # 回归结果
    regression_file = os.path.join(dataset_dir, 'model_evaluation_results.csv')
    if os.path.exists(regression_file):
        try:
            df = pd.read_csv(regression_file)
            results['regression'] = df.to_dict('records')
            print(f"✅ Loaded regression results for {dataset}")
        except Exception as e:
            results['regression'] = {'error': str(e)}
            print(f"❌ Error loading regression results for {dataset}: {e}")
    else:
        print(f"⚠️  No regression results file for {dataset}")
    
    # 分类结果
    classification_file = os.path.join(dataset_dir, 'tables', 'classification_performance.csv')
    if os.path.exists(classification_file):
        try:
            df = pd.read_csv(classification_file)
            results['classification'] = df.to_dict('records')
            print(f"✅ Loaded classification results for {dataset}")
        except Exception as e:
            results['classification'] = {'error': str(e)}
            print(f"❌ Error loading classification results for {dataset}: {e}")
    else:
        print(f"⚠️  No classification results file for {dataset}")
    
    return jsonify(results)

@app.route('/api/debug/files')
def debug_files():
    """调试API: 列出所有可用文件"""
    debug_info = {}
    
    for dataset in DATASETS:
        dataset_dir = f"outputs/{dataset}"
        debug_info[dataset] = {
            'directory_exists': os.path.exists(dataset_dir),
            'files': []
        }
        
        if os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, dataset_dir)
                    debug_info[dataset]['files'].append({
                        'relative_path': rel_path,
                        'full_path': full_path,
                        'size': os.path.getsize(full_path)
                    })
    
    return jsonify(debug_info)

@app.route('/api/project_summary')
def get_project_summary():
    """API: 获取项目总结信息"""
    summary_file = "outputs/project_summary.json"
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # 如果没有总结文件，生成基本信息
    basic_summary = {
        'datasets_available': [],
        'total_charts': 0,
        'total_models': 0
    }
    
    for dataset in DATASETS:
        dataset_dir = f"outputs/{dataset}"
        if os.path.exists(dataset_dir):
            basic_summary['datasets_available'].append(dataset)
            
            # 统计图表
            charts_dir = os.path.join(dataset_dir, 'charts')
            if os.path.exists(charts_dir):
                chart_count = len([f for f in os.listdir(charts_dir) if f.endswith('.png')])
                basic_summary['total_charts'] += chart_count
            
            # 统计模型
            models_dir = os.path.join(dataset_dir, 'models')
            if os.path.exists(models_dir):
                model_count = len([f for f in os.listdir(models_dir) if f.endswith('.pkl')])
                basic_summary['total_models'] += model_count
    
    return jsonify(basic_summary)

@app.errorhandler(404)
def not_found_error(error):
    """404错误处理"""
    return f"File not found", 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return f"Internal server error", 500

# 添加调试中间件
@app.before_request
def log_request_info():
    """记录请求信息用于调试"""
    if '/outputs/' in str(request.path) if 'request' in globals() else False:
        print(f"🌐 Request: {request.method} {request.path}")

if __name__ == '__main__':
    print("🌐 Starting Building Energy Prediction Dashboard...")
    print("📊 Individual Dataset Analysis Results")
    print("🔗 Available at: http://localhost:5000")
    print("📈 Dashboard: http://localhost:5000/dashboard")
    print("🔌 API endpoints:")
    print("   GET /api/datasets - List all datasets")
    print("   GET /api/dataset/<name>/results - Get dataset results")
    print("   GET /api/debug/files - Debug file listing")
    print("   GET /api/project_summary - Get project summary")
    print("\n💡 Press Ctrl+C to stop the server")
    
    # 启动时检查文件结构
    print("\n📁 Checking file structure...")
    for dataset in DATASETS:
        dataset_dir = f"outputs/{dataset}"
        exists = os.path.exists(dataset_dir)
        print(f"   {'✅' if exists else '❌'} {dataset_dir}")
        
        if exists:
            charts_dir = os.path.join(dataset_dir, 'charts')
            charts_exist = os.path.exists(charts_dir)
            print(f"      {'✅' if charts_exist else '❌'} charts/")
            
            if charts_exist:
                charts = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
                print(f"         📊 {len(charts)} chart files")
    
    app.run(debug=True, host='localhost', port=5000)