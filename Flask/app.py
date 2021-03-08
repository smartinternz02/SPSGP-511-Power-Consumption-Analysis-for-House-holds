import os
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,url_for
import pickle


app = Flask(__name__,template_folder=os.getcwd())
model = pickle.load(open('household_power_consumption.pkl', 'rb'))
#change your pickle here

@app.route('/')
def home():
    return redirect(url_for('postest'))

@app.route('/postest',methods=['GET'])
def postest():
    return render_template('Inoki.html')
    

@app.route('/postest',methods=['POST'])
def postestpost():
    print("SD")
    print(request.headers)
    #response = make_response(request.headers, 200)
    #response.mimetype = "text/plain"
    gar=request.form['gap']
    grp=request.form['grp']
    gi=request.form['gi']
    sr1=request.form['sr1']
    sr2=request.form['sr2']
    sr3=request.form['sr3']
    #x_test=[[gar,grp,gi,sr1,sr2,sr3]]
    x_test = [[float(x) for x in request.form.values()]]
    prediction = model.predict(x_test)
    #print(prediction)
    #output=prediction[0][0]
    
    return render_template('view.html',itext=prediction[0])



@app.route('/y_predict',methods=['GET','POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    
    
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0][0]
    return render_template('index.html', 
  prediction_text=
  'Compressive Strength of Concrete kg/m^3 {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)
