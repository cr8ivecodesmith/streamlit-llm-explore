import os
from pathlib import Path

import openai
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
PROJECT_ENV = PROJECT_DIR.joinpath('.env')

load_dotenv(PROJECT_ENV)


openai.organization = os.getenv('OPENAI_ORGANIZATION_ID')
openai.api_key = os.getenv('OPENAI_API_KEY')
models = openai.Model.list()

print(f'{models}')
