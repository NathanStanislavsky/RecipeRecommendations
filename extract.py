from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DATABASE")
INTERNAL_RATINGS_COLLECTION = os.getenv("MONGODB_REVIEWS_COLLECTION")
EXTERNAL_RATINGS_COLLECTION = os.getenv("MONGODB_EXTERNAL_REVIEWS_COLLECTION")

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
            print(f"Error getting records from collection: {error}")
            return None

    def get_combined_ratings_data(self, database_name, internal_collection_name, external_collection_name):
        print("--- Starting Data Extraction ---")

        # 1. Get internal ratings from MongoDB
        internal_ratings_list = self.get_all_records_from_collection(database_name, internal_collection_name)
        if internal_ratings_list is None:
            print("Failed to fetch internal ratings. Aborting.")
            return None
        internal_df = pd.DataFrame(internal_ratings_list)

        # 2. Load external ratings from MongoDB
        external_ratings_list = self.get_all_records_from_collection(database_name, external_collection_name)
        if external_ratings_list is None:
            print("Failed to fetch external ratings. Aborting.")
            return None
        external_df = pd.DataFrame(external_ratings_list)


        # 3. Combine the two DataFrames
        print("Combining internal and external ratings...")
        combined_df = pd.concat([internal_df, external_df], ignore_index=True)

        # 4. Final validation and cleanup
        required_columns = ['user_id', 'recipe_id', 'rating']
        if not all(col in combined_df.columns for col in required_columns):
            print(f"ERROR: Combined DataFrame is missing one of the required columns: {required_columns}")
            return None
        
        print(f"--- Data Extraction Complete. Total records: {len(combined_df)} ---")
        return combined_df[required_columns]


if __name__ == '__main__':
    extractor = Extract()
    if extractor.client:
        final_ratings_df = extractor.get_combined_ratings_data(
            database_name=DB_NAME,
            internal_collection_name=INTERNAL_RATINGS_COLLECTION,
            external_collection_name=EXTERNAL_RATINGS_COLLECTION
        )

        if final_ratings_df is not None:
            print("\nSuccessfully created combined DataFrame.")
            print("Top 5 rows:")
            print(final_ratings_df.head())
