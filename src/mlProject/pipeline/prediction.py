import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction


class CustomData:
    def __init__(self,
                 age:int,
                 sex:int,
                 cp:int,
                 trestbps:int,
                 chol:int,
                 fbs:int,
                 restecg:int,
                 thalach:int,
                 exang:int,
                 oldpeak:float,
                 slope:int,
                 ca:int,
                 thal:int):
        
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'age':[self.age],
                    'sex':[self.sex],
                    'cp':[self.cp],
                    'trestbps':[self.trestbps],
                    'chol':[self.chol],
                    'fbs':[self.fbs],
                    'restecg':[self.restecg],
                    'thalach':[self.thalach],
                    'exang':[self.exang],
                    'oldpeak':[self.oldpeak],
                    'slope':[self.slope],
                    'ca':[self.ca],
                    'thal':[self.thal]
                }
                df = pd.DataFrame(custom_data_input_dict)
                print(df)
                
                return df
            except Exception as e:
                
                raise(e)