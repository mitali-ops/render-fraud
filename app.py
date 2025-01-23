from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')  # HTML page for user input (in "templates" folder)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        form_data = request.form
        amount = float(form_data['amount'])
        old_balance_org = float(form_data['old_balance_org'])
        new_balance_orig = float(form_data['new_balance_orig'])
        old_balance_dest = float(form_data['old_balance_dest'])
        new_balance_dest = float(form_data['new_balance_dest'])

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'amount': [amount],
            'old_balance_org': [old_balance_org],
            'new_balance_orig': [new_balance_orig],
            'old_balance_dest': [old_balance_dest],
            'new_balance_dest': [new_balance_dest]
        })

        # Predict using the model
        prediction = model.predict(input_data)

        # Interpret prediction
        result = "Fraudulent" if prediction[0] == 1 else "Non-Fraudulent"

        # Return result
        return render_template('index.html', 
                               prediction_text=f"Transaction is: {result}")

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)})

# Main function
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
