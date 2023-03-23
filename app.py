from flask import Flask, render_template, request
import pickle
import numpy as np

# import sklearn
    
# Load the Logistic Regression model
classifier = pickle.load(open('finalmodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = float(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        bp = float(request.form['BloodPressure'])
        st = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])

        # int_features = [int(x) for x in request.form.values()]

        # final_features = [np.array(int_features)]

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        if my_prediction == 1:
            result = "Great! You DON'T have diabetes."
        if my_prediction == 0:
            result = "Oops! You Have Diabetes"

        return render_template('results.html', Prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)