import yaml,dill
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
import os,sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_yaml(file_path:str) ->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(file_path:str,content:object,replace:bool = False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
    
def save_numpy_array(file_path:str,array:np.array):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file:
            np.save(file,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)    
    
    
def save_object(file_path:str,obj:object) ->None:
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file:
           pickle.dump(obj,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)    
    
    
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(params.keys())[i]]
            print(list(models.keys())[i])
            gs = GridSearchCV(model,param_grid=param,n_jobs=-1)
            gs.fit(x_train,y_train)
            if hasattr(model,"set_param"):
                model.setattr(**gs.best_params_)
            
            model.fit(x_train,y_train)
            y_train_pred  = model.predict(x_train)
            y_test_pred  = model.predict(x_test)
            train_model_accuracy = accuracy_score(y_train,y_train_pred)
            test_model_accuracy = accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_accuracy
        return report
    
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e



            

                
            
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
