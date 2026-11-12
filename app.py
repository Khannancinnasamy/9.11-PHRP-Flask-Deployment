import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model (feature engineering + scaler + SVC are inside)
model = pickle.load(open("model/finalized_model_SVC.sav", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    Age = float(request.form["Age"])
    BMI = float(request.form["BMI"])
    Heart_Rate = float(request.form["Heart_Rate"])
    Follow_Up = float(request.form["Follow_Up"])
    Cholesterol_Status = request.form["Cholesterol_Status"]

    data = pd.DataFrame(
        [[Age, BMI, Heart_Rate, Follow_Up, Cholesterol_Status]],
        columns=["Age", "BMI", "Heart_Rate", "Follow_Up", "Cholesterol_Status"]
    )

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Disease Detected"
    else:
        result = "No Disease"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
