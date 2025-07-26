import os
import pandas as pd 
import json
from dotenv import load_dotenv
import redis
from surprise import SVD, Dataset, Reader

from extract import Extract

load_dotenv()
DB_NAME = os.getenv("MONGODB_DATABASE")
INTERNAL_RATINGS_COLLECTION = os.getenv("MONGODB_REVIEWS_COLLECTION")
EXTERNAL_RATINGS_COLLECTION = os.getenv("MONGODB_EXTERNAL_REVIEWS_COLLECTION")


class Train:
    def __init__(self, n_factors=100, n_epochs=20, random_state=42):
        self.algo = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            random_state=random_state,
            verbose=True,
        )
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            raise ValueError("FATAL: REDIS_URL not found in environment variables.")

        print("Found REDIS_URL. Connecting to Redis...")
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            print("Successfully connected to Redis.")
        except Exception as e:
            print(f"FATAL: Could not connect to Redis: {e}")
            raise

    def train_model(self, ratings_df):
        print("--- Training SVD Model ---")
        
        # Data validation and cleaning
        print(f"Original data shape: {ratings_df.shape}")
        
        # Remove rows with invalid user_id or recipe_id
        valid_data = ratings_df.dropna(subset=['user_id', 'recipe_id', 'rating'])
        print(f"After removing nulls: {valid_data.shape}")
        
        # Convert user_id and recipe_id to strings to avoid issues with numeric IDs
        valid_data['user_id'] = valid_data['user_id'].astype(str)
        valid_data['recipe_id'] = valid_data['recipe_id'].astype(str)
        
        # Remove any empty strings or problematic IDs
        valid_data = valid_data[
            (valid_data['user_id'] != '') & 
            (valid_data['user_id'] != '0') & 
            (valid_data['user_id'] != 'None') &
            (valid_data['recipe_id'] != '') & 
            (valid_data['recipe_id'] != '0') & 
            (valid_data['recipe_id'] != 'None')
        ]
        print(f"After filtering invalid IDs: {valid_data.shape}")
        
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            valid_data[["user_id", "recipe_id", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        self.algo.fit(trainset)
        print("--- Model Training Complete ---")
        return self.algo, trainset

    def extract_embeddings(self, algo, trainset):
        print("--- Extracting Embeddings ---")
        user_embeddings_raw = algo.pu
        recipe_embeddings_raw = algo.qi

        print(f"Raw user embeddings shape: {user_embeddings_raw.shape}")
        print(f"Raw recipe embeddings shape: {recipe_embeddings_raw.shape}")
        print(f"Number of users in trainset: {trainset.n_users}")
        print(f"Number of items in trainset: {trainset.n_items}")

        # Map internal surprise IDs back to application original IDs
        user_id_map = {}
        for inner_uid in range(trainset.n_users):
            try:
                original_uid = trainset.to_raw_uid(inner_uid)
                user_id_map[inner_uid] = original_uid
            except Exception as e:
                print(f"Warning: Could not map inner user ID {inner_uid} - {e}")
                continue
        
        recipe_id_map = {}
        for inner_iid in range(trainset.n_items):
            try:
                original_iid = trainset.to_raw_iid(inner_iid)
                recipe_id_map[inner_iid] = original_iid
            except Exception as e:
                print(f"Warning: Could not map inner item ID {inner_iid} - {e}")
                continue

        print(f"Mapped {len(user_id_map)} users")
        print(f"Mapped {len(recipe_id_map)} recipes")

        user_embeddings = {
            user_id_map[inner_id]: user_embeddings_raw[inner_id]
            for inner_id in user_id_map
        }
        recipe_embeddings = {
            recipe_id_map[inner_id]: recipe_embeddings_raw[inner_id]
            for inner_id in recipe_id_map
        }

        print(f"Extracted {len(user_embeddings)} user embeddings.")
        print(f"Extracted {len(recipe_embeddings)} recipe embeddings.")
        return user_embeddings, recipe_embeddings

    def save_to_redis(self, user_embeddings, recipe_embeddings):
        print("--- Saving Data to Redis ---")
        try:
            internal_user_embeddings = {
                uid: embed
                for uid, embed in user_embeddings.items()
                if not str(uid).startswith("ext_")
            }
            print(
                f"Filtered out external users. Saving {len(internal_user_embeddings)} internal user embeddings."
            )

            user_embeddings_serializable = {
                k: v.tolist() for k, v in internal_user_embeddings.items()
            }
            recipe_embeddings_serializable = {
                k: v.tolist() for k, v in recipe_embeddings.items()
            }

            self.redis_client.set(
                "user_embeddings", json.dumps(user_embeddings_serializable)
            )
            print("Successfully saved user_embeddings.")

            self.redis_client.set(
                "recipe_embeddings", json.dumps(recipe_embeddings_serializable)
            )
            print("Successfully saved recipe_embeddings.")

            print("--- All data saved to Redis successfully. ---")
        except Exception as e:
            print(f"ERROR: Failed to save data to Redis: {e}")


if __name__ == "__main__":
    # 1. Extract Data
    extractor = Extract()
    if not extractor.client:
        print("Could not connect to MongoDB. Exiting.")
        exit()

    ratings_df = extractor.get_combined_ratings_data(
        database_name=DB_NAME,
        internal_collection_name=INTERNAL_RATINGS_COLLECTION,
        external_collection_name=EXTERNAL_RATINGS_COLLECTION,
    )

    if ratings_df is None:
        print("Could not retrieve ratings data. Exiting.")
        exit()

    # 2. Train Model and Generate Data
    trainer = Train()
    algo, trainset = trainer.train_model(ratings_df)
    user_embeds, recipe_embeds = trainer.extract_embeddings(algo, trainset)

    # 3. Save Data to Production
    trainer.save_to_redis(user_embeds, recipe_embeds)
