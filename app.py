from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("insurance_model.pkl")

# Create the Flask app
app = Flask(__name__)

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route for form-based prediction (HTML form)
@app.route('/predict_web', methods=['POST'])
def predict_web():
    # Get data from form
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Format input as a DataFrame
    input_dict = {
        "sex": [sex],
        "smoker": [smoker],
        "region": [region],
        "age": [age],
        "bmi": [bmi],
        "children": [children]
    }
    input_df = pd.DataFrame(input_dict)

    # Predict
    prediction = model.predict(input_df)[0]

    return f"<h3>Predicted Insurance Charges: ₹{round(prediction, 2)}</h3>"

# Route for API-based prediction (JSON input)
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()

    # Format input as a DataFrame
    input_dict = {
        "sex": [data['sex']],
        "smoker": [data['smoker']],
        "region": [data['region']],
        "age": [data['age']],
        "bmi": [data['bmi']],
        "children": [data['children']]
    }
    input_df = pd.DataFrame(input_dict)

    # Predict
    prediction = model.predict(input_df)[0]

    return jsonify({'predicted_charges': round(prediction, 2)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
