import os
import sys
from network_security.exception.exception import NetworkSecurityException 
from network_security.logging.logger import logging
from network_security.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from network_security.entity.config_entity import ModelTrainerConfig
from network_security.utils.ml_utils.model.estimator import NetworkModel
from network_security.utils.main_utils.utils import save_object,load_object,load_numpy_array_data
from network_security.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,recall_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from network_security.utils.main_utils.utils import evaluate_model
import mlflow
# import dagshub
# dagshub.init(repo_owner='Divyanshu-Mathur', repo_name='Phish_Detect', mlflow=True)


class ModelTrainer :
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transform_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transform_artifact = data_transform_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
     
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score
            
            mlflow.log_metric("f1_score",f1_score) 
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model,"model")
     
     
     
    def train_model(self,x_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "KNN" :KNeighborsClassifier(),
                "XGB":XGBClassifier()
            }
        
        
        params={
            "Random Forest":{
                'criterion':['gini', 'entropy', 'log_loss'], 
                'max_features':['sqrt','log2',None],
                'n_estimators': [16,32,64,128],
                'max_depth':[2,4,6,8,9,10,12]
            },
            "Gradient Boosting":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.7,0.80,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['auto','sqrt','log2'],
                'n_estimators': [16,32,64,128]
            },
            
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                'splitter':['best','random'],
                'max_features':['sqrt','log2']
            },
            
            
            "Logistic Regression":{},
            
            "KNN" :{
                'n_neighbors':[1,2,3,4,5,6,7,8],
                'weights':['uniform', 'distance'],
                'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            
            "XGB":{}   
        }
        
        model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params = params)
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        logging.info(f"Best model is {best_model_name}")
        logging.info(f"Best score is {best_model_score}")
        
        best_model = models[best_model_name]
        y_train_pred = best_model.predict(x_train)
        classification_train_score = get_classification_score(y_pred=y_train_pred,y_true=y_train)
        self.track_mlflow(best_model,classification_train_score)
        
        y_test_pred = best_model.predict(x_test)
        classification_test_score = get_classification_score(y_pred=y_test_pred,y_true=y_test)
        self.track_mlflow(best_model,classification_test_score)
        
        preprocessor = load_object(file_path=self.data_transform_artifact.transformed_object_file_path)  
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        save_object( "final_model/model.pkl", best_model)
        
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_score,
                             test_metric_artifact=classification_test_score
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
     
        
    def init_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transform_artifact.transformed_train_file_path
            test_file_path = self.data_transform_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
            
            
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
