
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved model and feature names
model = joblib.load("loan_model.pkl")
features = joblib.load("loan_features.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()

        # Convert form values into numeric values
        encoding_maps = {
            "Gender": {"Female": 0, "Male": 1},
            "Married": {"No": 0, "Yes": 1},
            "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
            "Education": {"Graduate": 0, "Not Graduate": 1},
            "Self_Employed": {"No": 0, "Yes": 1},
            "Property_Area": {
                "Rural": 0,
                "Semiurban": 1,
                "Urban": 2
            }
        }

        input_values = {}

        for col, mapping in encoding_maps.items():
            input_values[col] = mapping[form_data[col]]

        # Convert numeric inputs
        applicant_income = float(form_data["ApplicantIncome"])
        coapplicant_income = float(form_data["CoapplicantIncome"])
        loan_amount = float(form_data["LoanAmount"])
        loan_term = float(form_data["Loan_Amount_Term"])
        credit_history = float(form_data["Credit_History"])

        # Apply the same log transformations as model training
        input_values["Credit_History"] = credit_history
        input_values["ApplicantIncomelog"] = np.log(applicant_income + 1)
        input_values["LoanAmountlog"] = np.log(loan_amount + 1)
        input_values["Loan_Amount_Term_log"] = np.log(loan_term + 1)

        total_income = applicant_income + coapplicant_income
        input_values["Total_Income_log"] = np.log(total_income + 1)

        # Arrange features in the exact training order
        input_data = pd.DataFrame([input_values])
        input_data = input_data.reindex(columns=features)

        # Check for missing or invalid feature values
        if input_data.isnull().any().any():
            raise ValueError("Some required input features are missing.")

        # Make prediction
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "Congratulations! Your loan may be approved."
        else:
            result = "Your loan may not be approved."

        return render_template(
            "index.html",
            prediction_text=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Prediction error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)