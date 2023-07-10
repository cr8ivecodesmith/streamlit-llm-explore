import os
import hashlib
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
PROJECT_ENV = PROJECT_DIR.joinpath('.env')
DB_PATH = PROJECT_DIR.joinpath('db')

load_dotenv(PROJECT_ENV)


if not DB_PATH.exists():
    DB_PATH.mkdir(mode=0o775, exist_ok=True)


def get_openai_api_key():
    return os.getenv('OPENAI_API_KEY')


def get_openai_org_id():
    return os.getenv('OPENAI_ORGANIZATION_ID')


def make_tempfile(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        file_path = Path(tf.name)

    with file_path.open('ab') as tf:
        for chunk in iter(lambda: uploaded_file.read(4096), b""):
            tf.write(chunk)
        uploaded_file.seek(0)

    return file_path


def compute_md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with file_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
