#loading the libraries
from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
import os

#initialising the flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/form')
def result():
    return render_template("form.html")

@app.route('/prediction',methods = ['post'])
def f2():
    if request.method == 'POST':
        results = request.form
        response = {}
        dic = {}

        for key,value in results.items():
            dic[key] = [value]

        df = pd.DataFrame.from_dict(dic)

        df.Gender.replace({'Male':1,'Female':0},inplace=True)
        df.Stream.replace({'Computer Science':0,'Information Technology':1,'Electronics And Communication':2,'Mechanical':3,'Electrical':4,'Civil':5},inplace=True)

        x = df

        scaler = pickle.load(open('scaler.pkl','rb'))
        svc = pickle.load(open('svc.pkl','rb'))

        x = pd.DataFrame(scaler.transform(x),columns=scaler.get_feature_names_out())


        y_p = svc.predict(x)
        response['y_p'] = y_p
        if(y_p):
            response['result'] = "Congratulations! You can get placed..."
            response['result_type'] = "positive"
        else:
            response['result'] = "Sorry! It seems the probability of getting placed is less for you..."
            response['result_type'] = "negative"

        return render_template('prediction.html',response=response)


if __name__ == '__main__':
    app.run(debug=True)