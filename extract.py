from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
MONGODB_URL = os.getenv("MONGODB_URL")

def connect_to_mongodb():
    try:
        client = MongoClient(MONGODB_URL)
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        return client
    except Exception as error:
        print(f"Error connecting to MongoDB: {error}")
        return None

class Extract:
    def __init__(self):
        self.client = connect_to_mongodb()

    def get_all_records_from_collection(self, database_name, collection_name):
        try:
            db = self.client[database_name]
            collection = db[collection_name]

            all_records = list(collection.find())
        
            print(f"Retrieved {len(all_records)} records from {collection_name}")
            return all_records
        except Exception as error:
            print(f"Error getting all records from collection: {error}")
            return None
    
