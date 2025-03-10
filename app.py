from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("rainfall_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the feature names
feature_names = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

@app.route('/')
def home():
    return render_template('index.html')  # Home Page (Form)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        input_features = [float(data[f]) for f in feature_names]
        input_array = np.array(input_features).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        result = "Rainfall" if prediction == 1 else "No Rainfall"

        return render_template('result.html', prediction=result)  # Navigate to result page

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")  # Error Handling

@app.route('/predict_again')
def predict_again():
    return render_template('index.html')  # Redirect to Home Page

if __name__ == '__main__':
    app.run(debug=True)
