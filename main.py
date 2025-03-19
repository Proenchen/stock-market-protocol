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

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')


if __name__ == "__main__": 
    app.run(debug=True)