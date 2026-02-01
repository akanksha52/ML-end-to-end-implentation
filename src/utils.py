import os
import sys
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from None
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report={}
        best_estimator={}
        for model_name, model in models.items():
            params=param[model_name]
            gs=GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)
            best_model=gs.best_estimator_
            y_test_pred=best_model.predict(X_test)
            test_model_score=r2_score(y_test, y_test_pred)
            report[model_name]=test_model_score
            best_estimator[model_name]=gs.best_estimator_
        return report, best_estimator
    except Exception as e:
        raise CustomException(e, sys)