from flask import Flask,render_template,request
# import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)  

model=pickle.load(open('model.pkl' , 'rb'))

@app.route('/')
def hello_world():
     return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    Gender=request.form.get('Gender')
    AGE=request.form.get('AGE')
    Urea=request.form.get('Urea')
    Cr=request.form.get('Cr')
    HbA1c=request.form.get('HbA1c')
    Chol=request.form.get('Chol')
    TG=request.form.get('TG')
    HDL=request.form.get('HDL')
    LDL=request.form.get('LDL')
    VLDL=request.form.get('VLDL')
    BMI=request.form.get('BMI')
    
    result = (model.predict([[Gender,AGE,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI]]))[0]
    
    if result == 1:
        return render_template('index.html', label=1)
    else:
        return render_template('index.html', label=-1)
    
    
    
    
    
    
    # if result == 1:
    #     return "Diabetic"
    # else:
    #     return "not diabetic"
    
    # return "{} {} {} {} {} {} {} {} {} {} {}".format(Gender, AGE,Urea,Cr,HbA1c,Chol,TG,HDL,LDL,VLDL,BMI)



if __name__ == "__main__":
    app.run(debug=True )