import numpy as np
import os
from sklearn.externals import joblib
from flask import Flask, render_template, request


BASE_DIR = 'D:/Study/DataScience/Flask/Application1'

#template_folder_path = os.path.join(BASE_DIR, 'templates')
#model_path = os.path.join(BASE_DIR, 'models/regression_model.pkl')

app = Flask(__name__)

model = joblib.load(open('models/regression_model.pkl', 'rb'))


# Default page api
@app.route('/')
def main():
    return render_template('home.html')


# Prediction page api
@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('home.html', prediction_text='Weight should be {} pounds.'.format(output))


if __name__ == "__main__":
    app.run()

