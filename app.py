
# fit scaler on training data
from sklearn.preprocessing import MinMaxScaler
import flask
import pickle
import pandas as pd
import werkzeug
import numpy as np
from flask import Flask, render_template, url_for, request


#load models at top of app to load into memory only one time
with open(r'models/model_eclf_new_.pkl', 'rb') as f:
    ensemble_model_new = pickle.load(f)

#load models at top of app to load into memory only one time

#Loading the dataset onto running environment 
df_train = pd.read_csv(r'data/data.csv')

#app is an object of home page using Flask package
app = flask.Flask(__name__,template_folder='templates')

#route for home page
@app.route('/')
def main():
    return (flask.render_template('index.html'))

#route for EDA_show page
@app.route('/EDA_show')
def report():
    return (flask.render_template('EDA_show.html'))

#route for predict page
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if flask.request.method == 'GET':
        return (flask.render_template('predict.html'))
    if flask.request.method == 'POST':
        col = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area']
        Dependents = float(flask.request.form['Dependents'])
        #print(Dependents)
        Gender = flask.request.form['Gender']
        #Gender = flask.request.form.get('Gender', type=int)
        #print(Gender)
        Loan_Amount_Term = flask.request.form['term']
        #print(Loan_Amount_Term)
        Married = flask.request.form['Married']
        #print(Married)
        Education = flask.request.form['Education']
        #print(Education)
        Self_Employed = flask.request.form['Self_Employed']
        #print(Self_Employed)
        ApplicantIncome = flask.request.form['ApplicantIncome']
        #print(ApplicantIncome)
        CoapplicantIncome = flask.request.form['CoapplicantIncome']
        Property_Area = flask.request.form['Property_Area']
        #print(Property_Area)
        Credit_History = flask.request.form['Credit_History']
        LoanAmount = flask.request.form['LoanAmount']
        #print(LoanAmount)
        #Data preprocessing
        if Property_Area == "Semiurban":
            Property_Area = 2
        elif Property_Area == "Urban":
            Property_Area = 1
        else:
            Property_Area = 0
        
        if Gender == "Female":
            Gender = 0
        else:
            Gender =1

        if Married == "Yes":
            Married = 1
        else:
            Married = 0

        if Education == "Graduate":
            Education = 1
        else:
            Education =0

        if Self_Employed == "Yes":
            Self_Employed = 1
        else:
            Self_Employed = 0
        

        #create an empty dataframe using pandas
        temp = pd.DataFrame(index=[1])
        #assign value into the dataframe
        temp["Gender"] = Gender
        temp["Married"] = Married
        temp["Dependents"] = Dependents
        temp["Education"] = Education
        temp["Self_Employed"] = Self_Employed
        temp["ApplicantIncome"] = ApplicantIncome
        temp["CoapplicantIncome"] = CoapplicantIncome
        temp["LoanAmount"] = LoanAmount
        temp["Loan_Amount_Term"] = Loan_Amount_Term
        temp["Credit_History"] = Credit_History
        temp["Property_Area"] = Property_Area
        print(temp)
        print(df_train)
        print(temp.columns)
        print(df_train.columns)
        # data normalization with sklearn
        # fit scaler on training data
        norm = MinMaxScaler()
        #norm = MinMaxScaler().fit(train)
        scale = temp.copy()
        scale = norm.fit_transform(temp)
        #make prediction
        scale = pd.DataFrame(scale, columns=temp.columns)
        pred = ensemble_model_new.predict(scale)
        print(pred)
        """Index(['Gender', 'Married', 'Dependents', 'Education', 'ApplicantIncome',
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
            'Property_Area'],
            dtype='object')
        Index(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'],
            dtype='object')"""
        if pred ==0:
            res = 'Loan Denied!'
        else:
            res = 'Congratulations! Loan Approved!'
        #render form again and add prediction)'''
        return flask.render_template(r'predict.html', result=res)
if __name__ == '__main__':
    app.run()
