from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline, CustomData


app = Flask(__name__) # initializing a flask app

# @app.route('/',methods=['GET'])  # route to display the home page
# def homePage():
#     return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            data = CustomData(
                age=request.form.get("age"),
                sex=request.form.get("sex"),
                cp=(request.form.get("cp")),
                trestbps=(request.form.get("trestbps")),
                chol=(request.form.get("chol")),
                fbs=request.form.get("fbs"),
                restecg=request.form.get("restecg"),
                thalach=(request.form.get("thalach")),
                exang=request.form.get("exang"),
                oldpeak=request.form.get("oldpeak"),
                slope=request.form.get("slope"),
                ca=request.form.get("ca"),
                thal=(request.form.get("thal"))
            )
       
         
            final_data = data.get_data_as_dataframe()
            # Make prediction
            predict_pipeline = PredictionPipeline()
            pred = predict_pipeline.predict(final_data)
            result = round(pred[0], 2)
            return render_template("result.html", final_result=result)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)