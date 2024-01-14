import os
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = join(dirname(__file__), '.env')
print(dotenv_path)

load_dotenv(dotenv_path)

ES_PASS = os.environ.get("ES_PASS")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

print(OPENAI_API_KEY)
print(ES_PASS)
