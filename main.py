from network_security.components.data_ingestion import DataIngestion
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
import os,sys
from network_security.entity.config_entity import ModelTrainerConfig
from network_security.components.model_trainer import ModelTrainer
import warnings
warnings.filterwarnings('ignore')




if __name__ == "__main__":
    try:
        
        training_pipeline = TrainingPipelineConfig()
        config = DataIngestionConfig(training_pipeline)
        ingestion = DataIngestion(config)
        logging.info("Started data Ingestion")
        artifacts = ingestion.init_data_ingestion()
        logging.info("Data ingestion Completed ")
        print(artifacts)
        data_valid_config = DataValidationConfig(training_pipeline)
        data_valid = DataValidation(artifacts,data_valid_config)
        logging.info("Data Validation initiated")
        data_valid_artifact = data_valid.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_valid_artifact)
        data_transformation_config=DataTransformationConfig(training_pipeline)
        logging.info("data Transformation started")
        data_transformation=DataTransformation(data_valid_artifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("data Transformation completed")
        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(training_pipeline)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transform_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.init_model_trainer()

        logging.info("Model Training artifact created")



    except Exception as e:
        raise NetworkSecurityException(e,sys)