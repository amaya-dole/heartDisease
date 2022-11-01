from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('heartDiseaseModel.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("heart_disease_predict.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.7):
        return render_template('heart_disease_predict.html',pred='Your Heart is in Danger.\nProbability of disease occuring is {}'.format(output),bhai="Your Heart is in Danger")
    elif output>str(0.4):
        return render_template('heart_disease_predict.html',pred='Your Heart is Moderately Safe. Take More Care.\n Probability of disease occuring is {}'.format(output),bhai="Your Heart is Moderately Safe for now. Take More Care.")
    else:
        return render_template('heart_disease_predict.html',pred='Your Heart is safe.\n Probability of disease occuring is {}'.format(output),bhai="Your Heart is Safe.")


if __name__ == '__main__':
    app.run(debug=True)
