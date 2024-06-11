import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Define the inverse_scaler function
X = pd.read_csv("salaries.csv").dropna()["SALARY"].astype("float64")
mu = X.mean()
sigma = X.std()
def inverse_scaler(x):
    return sigma * x + mu

# Create the Flask application
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("pipeline/model.pkl", "rb"))

@app.route('/')
def Home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST"])
def predict():
    inputs = list(request.form.values())
    numerical_input = inputs[0]
    categorical_input = inputs[1]
    article = np.where(categorical_input[0] in ["a", "e", "i", "o", "u"], "an", "a")
    designation = categorical_input.replace("-", " ")
    numerical_feature = [numerical_input]
    categorical_features = [categorical_input == x for x in ["analyst", "associate", "director", "manager", "senior_analyst", "senior_manager"]]
    features = [list(map(float, categorical_features + numerical_feature))]
    predicted_salary = model.predict(np.array(features))
    output = inverse_scaler(predicted_salary[0])
    return render_template("index.html", prediction_text = f"As {article} {designation} with {numerical_input} years of experience, your predicted salary is: {output:.2f} USD per year.")

if __name__ == "__main__":
    app.run(debug = True)