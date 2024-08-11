import os

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()

load_dotenv(dotenv_path)

HUGGING_TOKEN = os.getenv("HUGGING_TOKEN")