from flask import Flask, request,jsonify, render_template
import pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

#import ridge regressor and standard scaler pickle

ridge_model=pickle.load(open(r'E:\AI_Projects\Krish_Naik_ML\Linear_Regression\Car Fuel Efficiency Project\models\ridge.pkl','rb'))
standard_scaler=pickle.load(open(r'E:\AI_Projects\Krish_Naik_ML\Linear_Regression\Car Fuel Efficiency Project\models\scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        cylinders = int(request.form.get('cylinders'))
        displacement = float(request.form.get('displacement'))
        horsepower = float(request.form.get('horsepower'))
        weight = float(request.form.get('weight'))
        acceleration = float(request.form.get('acceleration'))
        model_year = int(request.form.get('model_year'))

        #input_data = np.array([cylinders,displacement,horsepower,weight,acceleration,model_year]).reshape(1,-1)
        input_data = pd.DataFrame([[cylinders, displacement, horsepower, weight, acceleration, model_year]],
                                  columns=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year'])
        new_scaled_data = standard_scaler.transform(input_data)
        result= ridge_model.predict(new_scaled_data)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(debug=True)  #0.0.0.0 is mapped to the local ip of the system