from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/outputs/charts/<filename>')
def chart_file(filename):
    return send_from_directory(os.path.join('outputs', 'charts'), filename)

if __name__ == '__main__':
    app.run(debug=True)