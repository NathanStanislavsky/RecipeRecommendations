from dotenv import load_dotenv
import os

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL")
MONGODB_URL = os.getenv("MONGODB_URL")







