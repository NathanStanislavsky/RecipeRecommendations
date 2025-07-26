import os
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
import redis
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

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
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["user_id", "recipe_id", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        self.algo.fit(trainset)
        print("--- Model Training Complete ---")
        return self.algo, trainset

    def extract_embeddings(self, algo, trainset):
        print("--- Extracting Embeddings ---")
        user_embeddings_raw = algo.pu
        recipe_embeddings_raw = algo.qi

        # Map internal surprise IDs back to application original IDs
        user_id_map = {trainset.to_inner_uid(uid): uid for uid in trainset.all_users()}
        recipe_id_map = {
            trainset.to_inner_iid(iid): iid for iid in trainset.all_items()
        }

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

    def calculate_similar_recipes(self, recipe_embeddings, top_n=10):
        print(f"--- Calculating Top {top_n} Similar Recipes ---")

        # Convert embeddings dict to a list of IDs and a numpy matrix for efficient calculation
        recipe_ids = list(recipe_embeddings.keys())
        embedding_matrix = np.array(list(recipe_embeddings.values()))

        # Calculate cosine similarity between all recipes
        cosine_sim_matrix = cosine_similarity(embedding_matrix)

        similar_recipes_map = {}
        for i, recipe_id in enumerate(recipe_ids):
            # Get similarity scores for the current recipe against all others
            sim_scores = list(enumerate(cosine_sim_matrix[i]))

            # Sort recipes based on similarity score, descending
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get the scores of the top_n+1 most similar recipes (the first one is the recipe itself)
            top_similar_indices = [score[0] for score in sim_scores[1 : top_n + 1]]

            # Map indices back to actual recipe IDs
            similar_recipes_map[recipe_id] = [
                recipe_ids[j] for j in top_similar_indices
            ]

        print("--- Similarity Calculation Complete ---")
        return similar_recipes_map

    def save_to_kv(self, user_embeddings, recipe_embeddings, similar_recipes):
        print("--- Saving Data to Vercel KV ---")
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

            self.redis_client.set("similar_recipes", json.dumps(similar_recipes))
            print("Successfully saved similar_recipes.")

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
    similar_recipes_map = trainer.calculate_similar_recipes(recipe_embeds)

    # 3. Save Data to Production
    trainer.save_to_kv(user_embeds, recipe_embeds, similar_recipes_map)
