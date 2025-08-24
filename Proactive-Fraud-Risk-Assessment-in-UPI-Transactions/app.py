import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template, redirect, session, url_for, flash
import logging
import joblib
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Model and Supporting Files ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'project_model.pkl')
    features_path = os.path.join(base_dir, 'feature_names.pkl')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')

    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    scaler = joblib.load(scaler_path)
    logging.info("Model, feature names, and scaler loaded successfully.")
except Exception as e:
    logging.error(f"FATAL ERROR: Could not load model files. {e}", exc_info=True)
    model = None
    scaler = None
    feature_names = None

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'a_very_strong_random_secret_key_12345' # A strong secret key is needed for sessions

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loggedin' not in session:
            flash('Please log in to access that page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Flask Routes ---

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('uname') == 'amitsingh' and request.form.get('pwd') == 'singh123':
            session['loggedin'] = True
            flash('Login Successful!', 'success')
            return redirect(url_for('upload'))
        else:
            # Provide specific feedback for a failed login
            flash('Invalid Credentials! Please try again.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

# Also, add a logout route if you haven't already
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    flash('You have been successfully logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/preview', methods=["POST"])
def preview():
    # Protect this page: only accessible if logged in
    if 'loggedin' in session:
        if 'datasetfile' in request.files and request.files['datasetfile'].filename != '':
            dataset_file = request.files['datasetfile']
            try:
                df = pd.read_csv(dataset_file)
                return render_template("preview.html", df_view=df)
            except Exception as e:
                flash(f'Error reading file: {e}', 'danger')
                return redirect(url_for('upload'))
        flash('No file selected.', 'warning')
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/prediction1')
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    # (chart route code remains the same)
    try:
        df = pd.read_csv('dataset/upi_fraud_dataset.csv')
        # ... (rest of your chart logic) ...
        fraud_counts = df['fraud_risk'].value_counts()
        pie_data = [['Type', 'Count'], ['Normal', int(fraud_counts.get(0,0))], ['Fraud', int(fraud_counts.get(1,0))]]
        
        category_map = {
            '0': 'Entertainment', '1': 'Food Dining', '2': 'Gas Transport', '3': 'Grocery NET',
            '4': 'Grocery POS', '5': 'Health Fitness', '6': 'Home', '7': 'Kids Pets',
            '8': 'Miscellaneous NET', '9': 'Miscellaneous POS', '10': 'Personal Care',
            '11': 'Shopping NET', '12': 'Shopping POS', '13': 'Travel'
        }
        category_counts = df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        bar_data = [['Category', 'Transactions']]
        for index, row in category_counts.iterrows():
            category_name = category_map.get(str(row['Category']), str(row['Category']))
            bar_data.append([category_name, int(row['Count'])])
            
        accuracy_data = [
            ["Algorithm", "Accuracy", { "role": "style" }], ["Logistic Regression", 88.5, "#DF01A5"],
            ["KNN", 92.1, "#DF01A5"], ["SVM", 89.0, "#DF01A5"], ["Naive Bayes", 85.4, "#DF01A5"],
            ["Decision Tree", 95.6, "#DF01A5"], ["Random Forest", 97.2, "#DF01A5"],
            ["CNN/ANN", 98.1, "#DF01A5"], ["XGBoost", 99.3, "gold"]
        ]
        return render_template('chart.html', pie_data=pie_data, bar_data=bar_data, accuracy_data=accuracy_data)
    except Exception as e:
        logging.error(f"Chart error: {e}")
        flash("Could not load chart data.", "error")
        return redirect(url_for('first'))


@app.route('/detect', methods=['POST'])
def detect():
    # (detect route code remains the same)
    try:
        trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))
        dob = pd.to_datetime(request.form.get("dob"))
        # ... (rest of your feature creation logic) ...
        trans_hour = trans_datetime.hour; trans_day = trans_datetime.day; trans_month = trans_datetime.month
        trans_year = trans_datetime.year; category = int(request.form.get("category")); amount = float(request.form.get("trans_amount"))
        state = int(request.form.get("state")); zip_code = int(request.form.get("zip")); trans_num = float(request.form.get("card_number"))
        age = np.round((trans_datetime - dob).days / 365.25)
        
        features = [trans_hour, trans_day, trans_month, trans_year, category, trans_num, age, amount, state, zip_code]
        
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)

        output = "Normal Transaction"
        reasons = []
        if prediction[0] == 1:
            output = "Fraudulent Transaction"
            importances = model.feature_importances_
            feature_importance_map = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            reasons = [feature[0].replace('_', ' ').title() for feature in feature_importance_map[:3]]
        return render_template('result.html', OUTPUT=output, REASONS=reasons)
    except Exception as e:
        flash(f"An error occurred: {e}", "danger")
        return redirect(url_for('prediction1'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


