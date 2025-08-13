import os
import uuid
import json
import traceback
from datetime import datetime, timezone
import base64
import numpy as np
from flask import request, jsonify
from werkzeug.utils import secure_filename

from celery_utils import celery_app
from supabase_client_setup import get_supabase_client
from config import (
    UPLOAD_FOLDER, RAW_EEG_BUCKET, REPORT_ASSET_BUCKET,
    DEFAULT_FS, ALZ_REF_PATH, NORM_REF_PATH
)
from utils import NpEncoder
from database import get_prediction_and_eeg, cleanup_storage_on_error
from ml_runner import run_model
from visualization import (
    generate_stacked_timeseries_image,
    generate_average_psd_image,
    generate_descriptive_stats
)
from similarity_analyzer import run_similarity_analysis
from pdf_generation import (
   TechnicalPDFReport, build_technical_pdf_report_content,
   PatientPDFReport, build_patient_pdf_report_content,
   ClinicianPDFReport, build_clinician_pdf_report_content
)
from routes import api_bp

def decode_base64_image_for_upload(base64_string):
    # Decode base64 image string for upload
    if not isinstance(base64_string, str): return None
    try: return base64.b64decode(base64_string.split(',', 1)[1])
    except (IndexError, TypeError, base64.binascii.Error): return None

@celery_app.task(name='predict_api.run_full_analysis_task')
def run_full_analysis_task(prediction_id, encoded_file_content, channel_index_for_plot, original_filename):
    # Main background task for running ML and generating reports
    supabase = get_supabase_client()
    asset_prefix = f"report_assets/{prediction_id}"
    report_generation_errors = []
    ml_output_file_path = None
    assets_to_clean = []
    temp_filename_in_worker = f"{prediction_id}_{original_filename}"
    temp_filepath_in_worker = os.path.join(UPLOAD_FOLDER, temp_filename_in_worker)
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Save EEG file locally
        with open(temp_filepath_in_worker, 'wb') as f:
            f.write(base64.b64decode(encoded_file_content))
        # Run ML model
        ml_output_file_path = run_model(temp_filepath_in_worker)
        if not os.path.exists(ml_output_file_path): raise FileNotFoundError(f"ML output at {ml_output_file_path} not found.")
        with open(ml_output_file_path, 'r') as f: ml_output_data = json.load(f)
        # Map majority prediction to human-readable label
        majority_pred = ml_output_data.get('majority_prediction')
        label_mapping = ml_output_data.get('label_mapping')
        if isinstance(label_mapping, dict) and majority_pred is not None:
            prediction_label = label_mapping.get(str(majority_pred), str(majority_pred))
        else:
            prediction_label = "Alzheimer's" if majority_pred == 1 else "Normal"
        consistency_metrics = ml_output_data.get('consistency_metrics')
        ml_update_payload = {
            "prediction": prediction_label, "probabilities": ml_output_data.get('probabilities'),
            "status": "Generating assets", "trial_predictions": ml_output_data.get('trial_predictions'),
            "consistency_metrics": consistency_metrics
        }
        # Update DB with ML results
        supabase.table('predictions').update(json.loads(json.dumps(ml_update_payload, cls=NpEncoder))).eq('id', prediction_id).execute()
        # Load EEG data for report
        prediction_data_for_report, eeg_data, error_msg = get_prediction_and_eeg(prediction_id)
        if error_msg or eeg_data is None: raise Exception(f"Could not load EEG data: {error_msg}")
        # Generate stats and plots
        stats_json = generate_descriptive_stats(eeg_data, DEFAULT_FS)
        ts_img_base64 = generate_stacked_timeseries_image(eeg_data, DEFAULT_FS)
        psd_img_base64 = generate_average_psd_image(eeg_data, DEFAULT_FS)
        similarity_results = run_similarity_analysis(temp_filepath_in_worker, ALZ_REF_PATH, NORM_REF_PATH, channel_index_for_plot)
        similarity_plot_base64 = similarity_results.get('plot_base64') if isinstance(similarity_results, dict) else None
        uploaded_asset_urls = {}
        # Upload generated images to storage
        for img_data, filename_s3, url_key in [
            (similarity_plot_base64, f"{asset_prefix}/similarity_plot.png", "similarity_plot_url"),
            (ts_img_base64, f"{asset_prefix}/timeseries.png", "timeseries_plot_url"),
            (psd_img_base64, f"{asset_prefix}/psd.png", "psd_plot_url")
        ]:
            assets_to_clean.append(filename_s3)
            img_bytes = decode_base64_image_for_upload(img_data)
            if img_bytes:
                try:
                    supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=filename_s3, file=img_bytes, file_options={"content-type": "image/png", "upsert": "true"})
                    uploaded_asset_urls[url_key] = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(filename_s3)
                except Exception as e: report_generation_errors.append(f"{url_key} Upload Fail")
        # Generate and upload PDF reports
        pdf_types = [
            ("technical", TechnicalPDFReport, build_technical_pdf_report_content),
            ("patient", PatientPDFReport, build_patient_pdf_report_content),
            ("clinician", ClinicianPDFReport, build_clinician_pdf_report_content)
        ]
        for pdf_type, PdfClass, builder in pdf_types:
            pdf_filename_s3 = f"{asset_prefix}/{pdf_type}_report.pdf"
            assets_to_clean.append(pdf_filename_s3)
            try:
                pdf_doc = PdfClass()
                pdf_doc.alias_nb_pages()
                builder_args = [pdf_doc, prediction_data_for_report, stats_json, similarity_results, consistency_metrics, ts_img_base64, psd_img_base64, similarity_plot_base64]
                if pdf_type == "patient":
                    builder_args = [pdf_doc, prediction_data_for_report, similarity_results, consistency_metrics, similarity_plot_base64]
                builder(*builder_args)
                pdf_bytes = bytes(pdf_doc.output())
                supabase.storage.from_(REPORT_ASSET_BUCKET).upload(path=pdf_filename_s3, file=pdf_bytes, file_options={"content-type": "application/pdf", "upsert": "true"})
                uploaded_asset_urls[f"{pdf_type}_pdf_url"] = supabase.storage.from_(REPORT_ASSET_BUCKET).get_public_url(pdf_filename_s3)
            except Exception as e_pdf:
                print(f"TASK ERROR [{prediction_id}]: PDF generation for {pdf_type} failed: {e_pdf}"); traceback.print_exc()
                report_generation_errors.append(f"{pdf_type} PDF Fail")
        # Final DB update with asset URLs and status
        final_status = "Completed" if not report_generation_errors else f"Completed with errors: {', '.join(report_generation_errors)}"
        final_update_payload = {"status": final_status, "stats_data": stats_json, "report_generated_at": datetime.now(timezone.utc).isoformat()}
        final_update_payload.update(uploaded_asset_urls)
        if isinstance(similarity_results, dict): final_update_payload["similarity_results"] = {k: v for k, v in similarity_results.items() if k != 'plot_base64'}
        supabase.table('predictions').update(json.loads(json.dumps(final_update_payload, cls=NpEncoder))).eq('id', prediction_id).execute()
    except Exception as e:
        # Handle errors, cleanup, and update DB
        print(f"!!! TASK ERROR [{prediction_id}]: A critical error occurred: {e}"); traceback.print_exc()
        supabase.table('predictions').update({"status": f"Failed: {str(e)[:100]}"}).eq('id', prediction_id).execute()
        for asset_path in assets_to_clean: cleanup_storage_on_error(REPORT_ASSET_BUCKET, asset_path)
    finally:
        # Remove temp files
        if os.path.exists(temp_filepath_in_worker):
            os.remove(temp_filepath_in_worker)
        if ml_output_file_path and os.path.exists(ml_output_file_path):
            os.remove(ml_output_file_path)

@api_bp.route('/predict', methods=['POST'])
def predict_route():
    # API endpoint for prediction request
    supabase = get_supabase_client()
    file = request.files.get('file')
    user_id = request.form.get('user_id')
    channel_index_str = request.form.get('channel_index', '0')
    try:
        channel_index_for_plot = int(channel_index_str)
    except (ValueError, TypeError):
        channel_index_for_plot = 0
    if not file or not user_id or not file.filename:
        return jsonify({'error': 'Invalid request: file and user_id are required.'}), 400
    prediction_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    temp_filepath = os.path.join(UPLOAD_FOLDER, f"{prediction_id}_{filename}")
    raw_eeg_storage_path = f'raw_eeg/{user_id}/{prediction_id}_{filename}'
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        # Save uploaded file temporarily
        file.save(temp_filepath)
        with open(temp_filepath, 'rb') as f:
            file_content = f.read()
            # Upload raw EEG to storage
            supabase.storage.from_(RAW_EEG_BUCKET).upload(path=raw_eeg_storage_path, file=file_content, file_options={"upsert": "true"})
        encoded_file_content = base64.b64encode(file_content).decode('utf-8')
        os.remove(temp_filepath)
        # Insert initial DB record
        initial_db_record = {
            "id": prediction_id, "user_id": user_id, "filename": filename,
            "status": "Pending", "prediction": "Processing...",
            "eeg_data_url": raw_eeg_storage_path
        }
        insert_res = supabase.table('predictions').insert(initial_db_record).execute()
        if hasattr(insert_res, 'error') and insert_res.error:
            raise Exception(f"DB insert failed: {insert_res.error.message}")
        # Start background analysis task
        run_full_analysis_task.delay(prediction_id, encoded_file_content, channel_index_for_plot, filename)
        return jsonify({"prediction_id": prediction_id}), 202
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        cleanup_storage_on_error(RAW_EEG_BUCKET, raw_eeg_storage_path)
        return jsonify({'error': f'Server error: {str(e)}'}), 500
