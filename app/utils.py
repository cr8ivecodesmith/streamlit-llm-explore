import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
PROJECT_ENV = PROJECT_DIR.joinpath('.env')

load_dotenv(PROJECT_ENV)


def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')


def get_openai_org_id():
    return os.getenv('OPENAI_ORGANIZATION_ID')
