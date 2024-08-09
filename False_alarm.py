from flask import Flask,request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_excel("C:\\Users\\USER\\Downloads\\False Alarm Cases.xlsx")


app = Flask(__name__)
#create an end point to train your model and save training data
# into the file

@app.route('/train_model')
def train():
    data = pd.read_excel("C:\\Users\\USER\\Downloads\\False Alarm Cases.xlsx")

    x = data.iloc[:,1:7]
    y=data['Spuriosity Index(0/1)']
    model = LogisticRegression()
    model.fit(x,y)
    joblib.dump(model,'train.pkl')
    return "Model Trained Successfully"

    # Load the pkl file and test our model we need to pass the next 
    # data via POST method
    # First we need to load our pickel file so that we can get the reference of the trainning data.

@app.route('/test_model',methods = ['POST'])
def test():
    pkl_file  = joblib.load('train.pkl')
    test_data = request.get_json()
    v1 = test_data['Ambient Temperature( deg C)']
    v2 = test_data['Calibration(days)']
    v3 = test_data['Unwanted substance deposition(0/1)']
    v4 = test_data['Humidity(%)']
    v5 = test_data['H2S Content(ppm)']
    v6 = test_data['detected by(% of sensors)']

    my_test_data = [v1,v2,v3,v4,v5,v6]
    my_data_array= np.array(my_test_data)
    test_array  = my_data_array.reshape(1,6)

    df = pd.DataFrame(test_array,columns=['Ambient Temperature( deg C)','Calibration(days)','Unwanted substance deposition(0/1)','Humidity(%)','H2S Content(ppm)','detected by(% of sensors)'])
    
    y_pred = pkl_file.predict(df)

    if y_pred ==1:
        return "False Alarm No Danger" 
    else:
        return "True Alarm, Danger"
    
    
    
app.run(port=5001)


