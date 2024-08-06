from flask import Flask, render_template, request, jsonify, redirect
from tensorflow.keras.models import load_model
import sqlite3
from flask import session


app = Flask(__name__)

app.secret_key = 'your_secret_key'  # Replace with a secret key for session management
app.config['SESSION_TYPE'] = 'filesystem'
# Load your trained deep learning model

model = load_model('model.h5')

# Database initialization
def init_db():
    conn = sqlite3.connect('your_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

import numpy as np

def preprocess_text(text):
    # Implement your preprocessing logic here
    # For example, lowercasing, removing special characters, etc.
    return text.lower()  # Replace with actual preprocessing

def detect_sql_injection(input_text):
    # Preprocess the input text
    preprocessed_input_text = preprocess_text(input_text)

    # Make a prediction using the loaded model
    prediction = model.predict(np.array([preprocessed_input_text]))

    # Check the model's prediction
    if prediction >= 0.5:  # Adjust the threshold based on your model's output
        return True  # SQL injection detected

    else:
        return False  # No SQL injection detected


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if detect_sql_injection(username) or detect_sql_injection(password):
            return render_template('register.html', sql_injection_detected=True)

        # Check if the username is already taken (you can use a SELECT query)
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            conn.close()
            return render_template('register.html', username_taken=True)

        # Insert the new user into the database (you can use an INSERT query)
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()

        # Redirect the user to the login page after successful registration
        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if detect_sql_injection(username) or detect_sql_injection(password):
            return render_template('login.html', sql_injection_detected=True)

        # Authenticate the user against the database
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Store user information in session for authentication
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect('/profile')
        else:
            return render_template('login.html', login_failed=True)

    return render_template('login.html')

from flask import session, redirect, url_for

@app.route('/profile')
def profile():
    if 'user_id' in session:
        username = session['username']
        return render_template('profile.html', username=username)
    else:
        return redirect('/login')
    
@app.route('/logout')
def logout():
    # Clear the session and redirect to the login page
    session.clear()
    return redirect('/login')


@app.route('/dashboard')
def dashboard():
    return render_template('admin/admin.html')

@app.route('/accounts')
def accounts():

    return render_template('admin/account.html')




import random

@app.route('/statistics')
def statistics():
    # Generate random data for the chart
    labels = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5"]
    data = [random.randint(10, 100) for _ in labels]

    return render_template('admin/statistics.html', labels=labels, data=data)


if __name__ == '__main__':
    # on public port 0000
    app.run(host='0.0.0.0', port=5000, debug=True)
