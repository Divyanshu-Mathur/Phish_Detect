import os,sys,json
from dotenv import load_dotenv

import certifi
import pandas as pd
import numpy as np
import pymongo
from network_security.logging.logger import logging
from network_security.exception.exception import NetworkSecurityException

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# print(MONGO_DB_URL)
ca = certifi.where()

class NetworkSecurityExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    
    def cv_json_converter(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_into_mongo(self,records,database,collection):
        try:
            self.records = records
            self.database = database
            self.collection = collection
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection  = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    

if __name__ =="__main__":
    FILE_PATH =  "Network_data\phisingData.csv" 
    Database = "mlproject"
    Collection = "networkdata"
    obj = NetworkSecurityExtract()
    records = obj.cv_json_converter(FILE_PATH)
    print(records)
    no_of_records = obj.insert_into_mongo(records=records,database=Database,collection=Collection)
    print(no_of_records)
      