import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase environment variables SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Application Constants
UPLOAD_FOLDER = 'uploads'
SIDDHI_FOLDER = 'SIDDHI'
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON_PATH = os.path.join(BACKEND_DIR, SIDDHI_FOLDER, 'output.json')

# Reference EEG data paths
ALZ_REF_PATH = os.path.join(BACKEND_DIR, 'feature_07.npy')
NORM_REF_PATH = os.path.join(BACKEND_DIR, 'feature_35.npy')

DEFAULT_FS = 128

# Supabase Storage Buckets
RAW_EEG_BUCKET = 'eeg-data'
REPORT_ASSET_BUCKET = 'report-assets'

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

os.makedirs(os.path.join(BACKEND_DIR, UPLOAD_FOLDER), exist_ok=True)