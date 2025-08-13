from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Supabase client initialized via supabase_client_setup.py.")

def get_supabase_client() -> Client:
    return supabase