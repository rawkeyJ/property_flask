#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import time
import overpy
from haversine import haversine, Unit 
import warnings
warnings.filterwarnings('ignore')





#Initialize the flask App
app = Flask(__name__, template_folder="templates", static_folder='static')

model11 = pickle.load(open('model_main_prop.pkl', 'rb'))

train_data = pd.read_csv('train_data_3_blore.csv')

X = train_data.drop(['price'],axis='columns')

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
        
    return model11.predict([x])[0]


@app.route('/')
def home():

    return render_template('index1.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    areaa = request.form.get("listing_area")
    loca = request.form.get("locality_name")
    bhk = request.form.get("tot_BHK")
    baths = request.form.get("tot_bath")

    prediction = predict_price(loca,areaa,baths,bhk)
    
    output = round(prediction,2)
    
    return render_template('index.html', prediction_text=output, loca=loca, baths=baths, bhk = bhk, areaa= areaa)

if __name__ == "__main__":
    app.run()
