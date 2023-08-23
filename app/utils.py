import os
import hashlib
import tempfile
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).parent.parent
PROJECT_ENV = PROJECT_DIR.joinpath('.env')

load_dotenv(PROJECT_ENV)

DB_PATH = PROJECT_DIR.joinpath('db')
DOCS_CACHE_PATH = PROJECT_DIR.joinpath('docs_cache')
HISTORY_PATH = PROJECT_DIR.joinpath('history')
PROMPT_CACHE_PATH = PROJECT_DIR.joinpath('prompt_cache')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


if not DB_PATH.exists():
    DB_PATH.mkdir(mode=0o775, exist_ok=True)

if not DOCS_CACHE_PATH.exists():
    DOCS_CACHE_PATH.mkdir(mode=0o775, exist_ok=True)

if not HISTORY_PATH.exists():
    HISTORY_PATH.mkdir(mode=0o775, exist_ok=True)

if not PROMPT_CACHE_PATH.exists():
    PROMPT_CACHE_PATH.mkdir(mode=0o775, exist_ok=True)


def make_tempfile(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        file_path = Path(tf.name)

    with file_path.open('ab') as tf:
        for chunk in iter(lambda: uploaded_file.read(4096), b""):
            tf.write(chunk)
        uploaded_file.seek(0)

    return file_path


def compute_checksum(val: str):
    hash_ = hashlib.sha1()
    hash_.update(bytes(val, 'utf-8'))
    return hash_.hexdigest()


def compute_file_checksum(file_path: Path):
    hash_ = hashlib.sha1()
    with file_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            hash_.update(chunk)
    return hash_.hexdigest()
