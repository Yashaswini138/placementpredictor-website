from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        features = [float(request.form[x]) for x in request.form]

        # Column names MUST match training
        columns = [
            "Gender",
            "Age",
            "Degree",
            "CGPA",
            "Internships_Count",
            "Projects_Count",
            "Certifications_Count",
            "Technical_Skills_Score_100",
            "Communication_Skills_Score_100",
            "Aptitude_Test_Score_100"
        ]

        # Convert to DataFrame
        input_df = pd.DataFrame([features], columns=columns)

        # Prediction
        prediction = model.predict(input_df)

        result = "Placed" if prediction[0] == 1 else "Not Placed"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)