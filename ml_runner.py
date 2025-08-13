import os
import sys
import subprocess
import traceback
from config import SIDDHI_FOLDER, OUTPUT_JSON_PATH

ML_RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
SIDDHI_PATH = os.path.join(ML_RUNNER_DIR, 'SIDDHI')
if SIDDHI_PATH not in sys.path:
    sys.path.insert(0, SIDDHI_PATH)

def run_model(filepath_to_process: str):
    """
    Runs the SIDDHI ML model script as a subprocess.
    """
    print(f"ML Runner: Executing ML model for: {filepath_to_process}")
    
    siddhi_absolute_path = SIDDHI_PATH
    absolute_filepath_for_ml = os.path.abspath(filepath_to_process)
    expected_output_json_in_siddhi = os.path.join(siddhi_absolute_path, 'output.json')

    if not os.path.isdir(siddhi_absolute_path):
        raise FileNotFoundError(f"SIDDHI directory not found at: {siddhi_absolute_path}")
    if not os.path.isfile(absolute_filepath_for_ml):
        raise FileNotFoundError(f"Input EEG file not found at: {absolute_filepath_for_ml}")

    if os.path.exists(expected_output_json_in_siddhi):
        try:
            os.remove(expected_output_json_in_siddhi)
            print(f"Removed existing output file: {expected_output_json_in_siddhi}")
        except Exception as rem_e:
            print(f"Warning: Could not remove {expected_output_json_in_siddhi}: {rem_e}")

    original_cwd = os.getcwd()
    print(f"Changing CWD from '{original_cwd}' to '{siddhi_absolute_path}'")
    os.chdir(siddhi_absolute_path)

    try:
        # Classification mode: 'binary' or 'multiclass'
        classification_mode = os.getenv('CLASSIFICATION_MODE', 'binary').strip().lower()
        if classification_mode not in ('binary', 'multiclass'):
            classification_mode = 'binary'

        if classification_mode == 'multiclass':
            # ADFD-Indep multi-class (0=CN,1=AD,2=MCI)
            cmd = [
                'python', 'run.py',
                '--task_name', 'classification',
                '--is_training', '0',
                '--model_id', 'ADFD-Indep',
                '--model', 'ADformer',
                '--data', 'ADFDIndep',
                '--e_layers', '6',
                '--batch_size', '128',
                '--d_model', '128',
                '--d_ff', '256',
                '--enc_in', '19',
                '--num_class', '3',
                '--seq_len', '96',
                '--input_file', absolute_filepath_for_ml,
                '--use_gpu', 'False',
                '--features', 'M',
                '--label_len', '48',
                '--pred_len', '96',
                '--n_heads', '8',
                '--d_layers', '1',
                '--factor', '1',
                '--embed', 'timeF',
                '--des', "'Exp'",
                '--seed', '41',
                '--patch_len_list', '2',
                '--up_dim_list', '19',
            ]
        else:
            # ADSZ-Indep binary (0=Normal,1=Alz)
            cmd = [
                'python', 'run.py', 
                '--task_name', 'classification', 
                '--is_training', '0', 
                '--model_id', 'ADSZ-Indep', 
                '--model', 'ADformer', 
                '--data', 'ADSZIndep', 
                '--e_layers', '6', 
                '--batch_size', '1',
                '--d_model', '128', 
                '--d_ff', '256', 
                '--enc_in', '19', 
                '--num_class', '2', 
                '--seq_len', '128', 
                '--input_file', absolute_filepath_for_ml,
                '--use_gpu', 'False',
                '--features', 'M', 
                '--label_len', '48',
                '--pred_len', '96', 
                '--n_heads', '8', 
                '--d_layers', '1', 
                '--factor', '1', 
                '--embed', 'timeF',
                '--des', "'Exp'",
                '--patch_len_list', '4',
                '--up_dim_list', '19',
            ]
        
        print(f"Running ML command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', timeout=360)
        
        print(f"ML Model STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"ML Model STDERR:\n{result.stderr}")
        
        if not os.path.exists('output.json'):
            raise FileNotFoundError(f"'output.json' not created in {siddhi_absolute_path} after script execution.")
        
        print("ML model script executed successfully.")
        return expected_output_json_in_siddhi

    except subprocess.CalledProcessError as proc_error:
        print(f"ML script execution failed (Return Code {proc_error.returncode})\n--- ML STDERR ---\n{proc_error.stderr}\n--- End ML STDERR ---")
        traceback.print_exc()
        raise
    except subprocess.TimeoutExpired:
        print("ML script execution timed out.")
        raise TimeoutError("ML model execution timed out.")
    except FileNotFoundError as fnf_error:
        print(f"File System Error: {fnf_error}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        raise
    finally:
        print(f"Changing CWD back to original: {original_cwd}")
        os.chdir(original_cwd)
