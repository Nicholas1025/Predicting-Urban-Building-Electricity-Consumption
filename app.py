from flask import Flask, render_template, send_from_directory, abort
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/outputs/<path:filename>')
def outputs_file(filename):
    """Serve files from outputs directory"""
    try:
        return send_from_directory('outputs', filename)
    except FileNotFoundError:
        abort(404)

@app.route('/static/<path:filename>')
def static_file(filename):
    """Serve static files"""
    try:
        return send_from_directory('static', filename)
    except FileNotFoundError:
        abort(404)

@app.errorhandler(404)
def not_found(error):
    return f"File not found: {error}", 404

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)