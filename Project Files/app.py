from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    age_first_funding_year = float(request.form['age_first_funding_year'])
    age_last_funding_year = float(request.form['age_last_funding_year'])
    age_first_milestone_year = float(request.form['age_first_milestone_year'])
    age_last_milestone_year = float(request.form['age_last_milestone_year'])
    relationships = float(request.form['relationships'])
    funding_rounds = float(request.form['funding_rounds'])
    funding_total_usd = float(request.form['funding_total_usd'])
    milestones = float(request.form['milestones'])
    avg_participants = float(request.form['avg_participants'])

    # Build DataFrame with same column names as training
    input_data = pd.DataFrame([[
        age_first_funding_year,
        age_last_funding_year,
        age_first_milestone_year,
        age_last_milestone_year,
        relationships,
        funding_rounds,
        funding_total_usd,
        milestones,
        avg_participants
    ]], columns=[
        'age_first_funding_year',
        'age_last_funding_year',
        'age_first_milestone_year',
        'age_last_milestone_year',
        'relationships',
        'funding_rounds',
        'funding_total_usd',
        'milestones',
        'avg_participants'
    ])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction] * 100

    result = "Acquired" if prediction == 1 else "Closed"
    confidence = f"{probability:.2f}%"

    return render_template('result.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
