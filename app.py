import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

appian = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@appian.route('/')
def home():
    return render_template('index.html')

@appian.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #print("################",final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    appian.run(debug=True)