import io
import numpy as np
import traceback
from supabase_client_setup import get_supabase_client
from config import RAW_EEG_BUCKET

def get_prediction_and_eeg(prediction_id: str):
    """
    Fetches a prediction record and its associated raw EEG data from Supabase.
    """
    supabase = get_supabase_client()
    print(f"DB Helper: Fetching record for prediction ID: {prediction_id}")
    prediction_rec = None
    try:
        prediction_res = supabase.table('predictions').select('*').eq('id', prediction_id).maybe_single().execute()
        
        if not prediction_res.data:
            return None, None, "Prediction record not found"
        
        prediction_rec = prediction_res.data
        eeg_url_path = prediction_rec.get('eeg_data_url')

        if not eeg_url_path:
            return prediction_rec, None, "EEG data URL missing from prediction record"

        print(f"DB Helper: Downloading EEG data from path: {eeg_url_path}")
        eeg_file_response = supabase.storage.from_(RAW_EEG_BUCKET).download(eeg_url_path)

        if not isinstance(eeg_file_response, bytes):
            error_message = f"Failed to download raw EEG file. Response: {getattr(eeg_file_response, 'message', str(eeg_file_response))}"
            print(f"DB Helper Error: {error_message}")
            return prediction_rec, None, error_message

        with io.BytesIO(eeg_file_response) as f:
            eeg_data = np.load(f, allow_pickle=True)
        
        # Standardize EEG data shape (samples, channels)
        if eeg_data.ndim == 3: 
            print(f"DB Helper: Original 3D EEG data shape: {eeg_data.shape}. Using first trial.")
            eeg_data = eeg_data[0, :, :] 
        
        if eeg_data.ndim != 2:
            raise ValueError(f"Unsupported EEG data dimension after potential trial selection: {eeg_data.ndim}")

        if eeg_data.shape[0] < eeg_data.shape[1]:
            print(f"DB Helper: Transposing EEG data from {eeg_data.shape} to {(eeg_data.shape[1], eeg_data.shape[0])}")
            eeg_data = eeg_data.T
            
        if eeg_data.ndim != 2: 
             raise ValueError(f"Final EEG data is not 2D after processing: {eeg_data.shape}")

        print(f"DB Helper: Successfully processed EEG data. Final shape: {eeg_data.shape}")
        return prediction_rec, eeg_data.astype(np.double), None

    except Exception as e:
        print(f"DB Helper Error for prediction ID {prediction_id}: {e}")
        traceback.print_exc()
        return (prediction_rec if prediction_rec else None), None, f"Error accessing/processing data: {str(e)}"

def cleanup_storage_on_error(bucket_name: str, path: str):
    """
    Removes an object from the specified Supabase storage bucket.
    """
    supabase = get_supabase_client()
    try:
        if bucket_name and path:
            print(f"Storage Cleanup: Attempting to remove '{path}' from bucket '{bucket_name}'")
            response = supabase.storage.from_(bucket_name).remove([path])
            # print(f"Storage Cleanup Response: {response}") # For detailed debugging
    except Exception as e:
        print(f"Error during storage cleanup of '{path}' in '{bucket_name}': {e}")
        traceback.print_exc()