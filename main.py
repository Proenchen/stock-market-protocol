from flask import Flask, render_template

app: Flask = Flask(__name__)

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route("/")
def index():
    return render_template('home.html')

if __name__ == "__main__": 
    app.run(debug=True)