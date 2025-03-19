from flask import Flask, render_template, redirect, url_for

app: Flask = Flask(__name__)

@app.route("/")
def index():
    return redirect(url_for('home'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/setup')
def setup():
    return render_template('setup.html')

@app.route('/usage')
def usage():
    return render_template('usage.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')


if __name__ == "__main__": 
    app.run(debug=True)