import flask
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
os.chdir(r"C:\Users\Administrator\Downloads\loan-default-prediction-master")
#load models at top of app to load into memory only one time

#load models at top of app to load into memory only one time for ease of saving time.
with open('models/xgb_final.pkl', 'rb') as f:
    xgb_final = pickle.load(f)
#load models at top of app to load into memory only one time
with open('models/random_final.pkl', 'rb') as f:
    random_forest = pickle.load(f)

ss = StandardScaler()

#feature space dataset
df_train_jl_scale = pd.read_csv('data/df_train_scaled.csv')


drop_columns=["Loan_ID"]
#data preprocessing 


app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return (flask.render_template('index.html'))


@app.route('/report')
def report():
    return (flask.render_template('report.html'))


@app.route('/jointreport')
def jointreport():
    return (flask.render_template('jointreport.html'))
