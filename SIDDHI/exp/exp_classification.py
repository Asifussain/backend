# backend/SIDDHI/exp/exp_classification.py
from copy import deepcopy
from .exp_basic import Exp_Basic
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import Counter
import json
import scipy.stats
import traceback

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from models.ADformer import Model as ADformer_Model
except ImportError as e:
    print(f"Warning: Could not import default model classes: {e}. Ensure models are discoverable.")
    class PlaceholderModel:
        class Model:
            def __init__(self, args): pass
    ADformer_Model = PlaceholderModel.Model

warnings.filterwarnings("ignore")

# JSON encoder for numpy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj): return None
            if np.isinf(obj): return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        # SWA initialization
        if not hasattr(args, 'swa') or not args.swa:
            self.swa = False
            self.swa_model = None
        else:
            if self.model is not None:
                self.swa = args.swa
                self.swa_model = optim.swa_utils.AveragedModel(self.model)
                print("SWA model wrapper initialized.")
            else:
                print("Warning: SWA enabled but self.model is not yet built. SWA wrapper not initialized.")
                self.swa = False
                self.swa_model = None

    def _build_model(self):
        print("Building model...")
        model_dict = {
            'ADformer': ADformer_Model,
        }
        model_name = getattr(self.args, 'model', None)
        if model_name is None or model_name not in model_dict:
            raise ValueError(f"Model name '{model_name}' not found in model_dict or args. Available: {list(model_dict.keys())}")

        print(f"Using model class: {model_dict[model_name]}")
        model = model_dict[model_name](self.args).float()

        use_gpu = hasattr(self.args, 'use_gpu') and self.args.use_gpu
        use_multi_gpu = hasattr(self.args, 'use_multi_gpu') and self.args.use_multi_gpu
        device_ids = getattr(self.args, 'device_ids', [])

        if use_multi_gpu and use_gpu and device_ids and torch.cuda.device_count() > 1:
            try:
                valid_device_ids = [int(d) for d in device_ids]
                model = nn.DataParallel(model, device_ids=valid_device_ids)
                print(f"Using DataParallel on devices: {valid_device_ids}")
            except ValueError:
                 print(f"Warning: Invalid device_ids format: {device_ids}. Using default device instead.")
            except AssertionError as e:
                 print(f"Warning: DataParallel assertion error: {e}. Using default device instead.")

        print("Model built successfully.")
        return model

    def predict_unlabeled_sample(self, npy_file_path, setting, device):
        """
        Loads the trained model checkpoint and predicts the class(es) for an unlabeled NPY file.
        """
        # Build checkpoint path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not setting or not isinstance(setting, str):
            print("Error: 'setting' argument is missing or invalid.")
            raise ValueError("Missing 'setting' argument required to locate the correct checkpoint directory.")

        relative_checkpoint_dir_path = os.path.join(
            '..',
            'checkpoints',
            self.args.task_name,
            self.args.model_id,
            self.args.model,
            setting
        )
        model_path = os.path.join(script_dir, relative_checkpoint_dir_path, 'checkpoint.pth')
        model_path = os.path.normpath(model_path)

        print(f"Looking for model checkpoint at calculated path: {model_path}")

        if not os.path.exists(model_path):
            print(f"Warning: Model checkpoint not found at expected location.")
            print(f"Expected location: {model_path}")
            print(f"Current working directory: {os.getcwd()}")
            # Try a resilient search inside the model directory for a close match
            try:
                base_model_dir = os.path.normpath(os.path.join(script_dir, '..', 'checkpoints', self.args.task_name, self.args.model_id, self.args.model))
                if os.path.isdir(base_model_dir):
                    print(f"Searching for fallback checkpoints under: {base_model_dir}")
                    candidates = []
                    expected_prefix = setting
                    # If the setting contains a seed suffix, allow prefix match without seed
                    seedless_prefix = setting.split('_seed')[0]
                    for name in os.listdir(base_model_dir):
                        subdir = os.path.join(base_model_dir, name)
                        if not os.path.isdir(subdir):
                            continue
                        ckpt_candidate = os.path.join(subdir, 'checkpoint.pth')
                        if os.path.exists(ckpt_candidate):
                            starts_ok = name.startswith(expected_prefix) or name.startswith(seedless_prefix)
                            candidates.append((ckpt_candidate, starts_ok, os.path.getmtime(ckpt_candidate)))
                    # Prefer prefix matches; then latest by mtime
                    if candidates:
                        candidates.sort(key=lambda x: (0 if x[1] else 1, -x[2]))
                        model_path = candidates[0][0]
                        print(f"Using fallback checkpoint: {model_path}")
                    else:
                        print("No fallback checkpoints found.")
                else:
                    print(f"Base model directory does not exist: {base_model_dir}")
            except Exception as search_err:
                print(f"Error while searching for fallback checkpoints: {search_err}")

            if not os.path.exists(model_path):
                parent_dir = os.path.dirname(model_path)
                if not os.path.exists(parent_dir):
                    print(f"Parent directory '{parent_dir}' also does not exist.")
                    print("Verify arguments: task_name, model_id, model, and the passed 'setting' string.")
                else:
                    print(f"Parent directory '{parent_dir}' exists, but 'checkpoint.pth' is missing.")
                raise FileNotFoundError("Model checkpoint not found at calculated or fallback path: %s" % model_path)

        # Load model weights
        print(f"Loading model state from {model_path} onto device: {device}")
        use_swa = hasattr(self, 'swa') and self.swa and self.swa_model is not None

        try:
            # Inspect checkpoint to align model architecture if possible
            try:
                raw_for_shape = torch.load(model_path, map_location='cpu')
                state_like = raw_for_shape
                if isinstance(raw_for_shape, dict):
                    for key in ['state_dict', 'model_state_dict', 'model']:
                        if key in raw_for_shape and isinstance(raw_for_shape[key], dict):
                            state_like = raw_for_shape[key]
                            break
                if isinstance(state_like, dict):
                    # Normalize keys (strip DataParallel prefix) for architecture inference
                    state_scan = {}
                    for k, v in state_like.items():
                        nk = k
                        if nk.startswith('module.'):
                            nk = nk.replace('module.', '', 1)
                        state_scan[nk] = v
                    # Infer seq_len from fc weights in channel embedding
                    seq_len_inferred = None
                    d_model_inferred = None
                    patch_len_list_inferred = []
                    up_dim_list_inferred = []
                    # d_model from temporal tokenConv out_channels
                    for k, v in state_scan.items():
                        if 'enc_embedding.value_embeddings_t.0.tokenConv.weight' in k and isinstance(v, torch.Tensor):
                            d_model_inferred = int(v.shape[0])
                            break
                    # seq_len and up_dim list
                    for k, v in state_scan.items():
                        if 'enc_embedding.value_embeddings_c.0.fc.weight' in k and isinstance(v, torch.Tensor):
                            # shape [d_model, seq_len]
                            seq_len_inferred = int(v.shape[1])
                            break
                    # patch_len_list from temporal tokenConv kernel width
                    i = 0
                    while f'enc_embedding.value_embeddings_t.{i}.tokenConv.weight' in state_scan:
                        w = state_scan[f'enc_embedding.value_embeddings_t.{i}.tokenConv.weight']
                        if isinstance(w, torch.Tensor) and w.ndim == 4:
                            patch_len_list_inferred.append(int(w.shape[3]))
                        i += 1
                    # up_dim_list from channel tokenConv out_channels
                    j = 0
                    while f'enc_embedding.value_embeddings_c.{j}.tokenConv.weight' in state_scan:
                        w = state_scan[f'enc_embedding.value_embeddings_c.{j}.tokenConv.weight']
                        if isinstance(w, torch.Tensor) and w.ndim == 3:
                            up_dim_list_inferred.append(int(w.shape[0]))
                        j += 1
                    # e_layers from encoder layer count
                    max_layer_idx = -1
                    for k in state_scan.keys():
                        if k.startswith('encoder.attn_layers.'):
                            try:
                                idx = int(k.split('.')[2])
                                if idx > max_layer_idx:
                                    max_layer_idx = idx
                            except Exception:
                                pass
                    e_layers_inferred = max_layer_idx + 1 if max_layer_idx >= 0 else None

                    args_changed = False
                    if seq_len_inferred and getattr(self.args, 'seq_len', None) != seq_len_inferred:
                        print(f"Adjusting seq_len from {getattr(self.args,'seq_len',None)} to {seq_len_inferred} based on checkpoint")
                        self.args.seq_len = seq_len_inferred
                        args_changed = True
                    if d_model_inferred and getattr(self.args, 'd_model', None) != d_model_inferred:
                        print(f"Adjusting d_model from {getattr(self.args,'d_model',None)} to {d_model_inferred} based on checkpoint")
                        self.args.d_model = d_model_inferred
                        args_changed = True
                    if patch_len_list_inferred and getattr(self.args, 'patch_len_list', None) != ','.join(map(str, patch_len_list_inferred)):
                        self.args.patch_len_list = ','.join(map(str, patch_len_list_inferred))
                        print(f"Adjusting patch_len_list to {self.args.patch_len_list} based on checkpoint")
                        args_changed = True
                    if up_dim_list_inferred and getattr(self.args, 'up_dim_list', None) != ','.join(map(str, up_dim_list_inferred)):
                        self.args.up_dim_list = ','.join(map(str, up_dim_list_inferred))
                        print(f"Adjusting up_dim_list to {self.args.up_dim_list} based on checkpoint")
                        args_changed = True
                    if e_layers_inferred and getattr(self.args, 'e_layers', None) != e_layers_inferred:
                        print(f"Adjusting e_layers from {getattr(self.args,'e_layers',None)} to {e_layers_inferred} based on checkpoint")
                        self.args.e_layers = e_layers_inferred
                        args_changed = True
                    # Derive feature toggles
                    self.args.no_temporal_block = len(patch_len_list_inferred) == 0
                    self.args.no_channel_block = len(up_dim_list_inferred) == 0
                    # If architecture changed, rebuild model
                    if args_changed:
                        print("Rebuilding model to match checkpoint architecture...")
                        self.model = self._build_model().to(device)
                        if use_swa and self.swa_model is not None:
                            self.swa_model = optim.swa_utils.AveragedModel(self.model)
                    # Log effective architecture
                    try:
                        print(f"Effective architecture → seq_len={self.args.seq_len}, d_model={self.args.d_model}, e_layers={self.args.e_layers}, patch_len_list={self.args.patch_len_list}, up_dim_list={self.args.up_dim_list}")
                    except Exception:
                        pass
            except Exception as arch_infer_err:
                print(f"Could not infer architecture from checkpoint: {arch_infer_err}")

            def _load_state_dict_safely(target_model, checkpoint_path, device):
                raw = torch.load(checkpoint_path, map_location=device)
                state_dict_candidate = raw
                if isinstance(raw, dict):
                    for key in ['state_dict', 'model_state_dict', 'model']:
                        if key in raw and isinstance(raw[key], dict):
                            state_dict_candidate = raw[key]
                            break
                # Strip DataParallel 'module.' prefix if present
                if isinstance(state_dict_candidate, dict) and any(k.startswith('module.') for k in state_dict_candidate.keys()):
                    state_dict_candidate = {k.replace('module.', '', 1): v for k, v in state_dict_candidate.items()}
                # First attempt strict load, then fallback to non-strict
                try:
                    missing, unexpected = target_model.load_state_dict(state_dict_candidate, strict=True)
                    if isinstance(missing, list) and isinstance(unexpected, list) and (missing or unexpected):
                        print(f"Warning: Strict load reported missing={len(missing)}, unexpected={len(unexpected)}. Proceeding.")
                except Exception as strict_err:
                    print(f"Strict load_state_dict failed: {strict_err}. Trying non-strict load.")
                    try:
                        res = target_model.load_state_dict(state_dict_candidate, strict=False)
                        if hasattr(res, 'missing_keys') or hasattr(res, 'unexpected_keys'):
                            print(f"Non-strict load: missing_keys={getattr(res,'missing_keys',[])}, unexpected_keys={getattr(res,'unexpected_keys',[])}")
                    except Exception as non_strict_err:
                        print(f"Non-strict load failed due to size mismatches: {non_strict_err}. Filtering compatible keys and retrying.")
                        model_sd = target_model.state_dict()
                        filtered = {k: v for k, v in state_dict_candidate.items() if k in model_sd and getattr(model_sd[k], 'shape', None) == getattr(v, 'shape', None)}
                        dropped = [k for k in state_dict_candidate.keys() if k not in filtered]
                        print(f"Filtered checkpoint params: kept={len(filtered)}, dropped={len(dropped)} due to shape/key mismatch.")
                        target_model.load_state_dict(filtered, strict=False)

            if use_swa:
                if self.swa_model is None:
                     raise RuntimeError("SWA is enabled but swa_model is None. Cannot load weights.")
                _load_state_dict_safely(self.swa_model, model_path, device)
                self.swa_model = self.swa_model.to(device)
                self.swa_model.eval()
                model_to_use = self.swa_model
                print("Using SWA model weights for prediction.")
            else:
                if self.model is None:
                    raise RuntimeError("Base model (self.model) is None. Cannot load weights.")
                _load_state_dict_safely(self.model, model_path, device)
                self.model.eval()
                model_to_use = self.model
                print("Using standard model weights for prediction.")
        except Exception as load_err:
            print(f"Error loading model state dict from {model_path}: {load_err}")
            traceback.print_exc()
            raise

        # Load input data
        try:
            print(f"Loading input EEG data from: {npy_file_path}")
            X_orig = np.load(npy_file_path, allow_pickle=True)
            print(f"Original input data shape: {X_orig.shape}")
        except Exception as e:
            print(f"Error loading .npy file {npy_file_path}: {e}")
            traceback.print_exc()
            raise

        # Handle input shape
        if X_orig.ndim == 3:
            num_trials, seq_len_data, channels_data = X_orig.shape
            print(f"Input is 3D ({X_orig.shape}), processing {num_trials} trials/segments.")
            X_batch = X_orig
        elif X_orig.ndim == 2:
            num_trials = 1
            seq_len_data, channels_data = X_orig.shape
            print(f"Input is 2D ({X_orig.shape}), processing as 1 trial/segment.")
            X_batch = np.expand_dims(X_orig, axis=0)
        else:
            raise ValueError(f"Unsupported input data dimension: {X_orig.ndim}. Expected 2 or 3.")

        # Validate shape
        expected_seq_len = getattr(self.args, 'seq_len', None)
        expected_channels = getattr(self.args, 'enc_in', None)

        if expected_seq_len is None or expected_channels is None:
             print("Warning: Model's expected seq_len or enc_in not found in args. Skipping shape validation.")
        else:
            if seq_len_data != expected_seq_len:
                print(f"Warning: Input sequence length {seq_len_data} != expected {expected_seq_len}.")
            if channels_data != expected_channels:
                raise ValueError(f"Input data has {channels_data} channels, model expects {expected_channels}.")

        print(f"Data shape for model processing: {X_batch.shape}")

        # Convert to tensor
        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)

        # Inference
        all_predictions = []
        all_probabilities = []
        num_classes = None
        print(f"Running inference on {num_trials} trial(s) using device: {device}...")

        try:
            with torch.no_grad():
                batch_size, current_seq_len, _ = X_tensor.shape
                padding_mask = torch.ones((batch_size, current_seq_len), dtype=torch.bool).to(device)

                outputs = model_to_use(X_tensor, padding_mask, None, None)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # If outputs are sequence logits, average over sequence length
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)  # shape -> [batch_size, num_classes]

                probs = torch.nn.functional.softmax(outputs, dim=-1)
                num_classes = probs.shape[-1]

                predictions = torch.argmax(probs, dim=-1)  # one prediction per trial
                all_predictions = predictions.cpu().numpy()
                all_probabilities = probs.cpu().numpy()

                try:
                    avg_probs = np.mean(all_probabilities, axis=0).tolist()  # average per class
                    print(f"Average class probabilities across trials: {avg_probs}")
                except Exception:
                    pass

        except Exception as inference_err:
            print(f"Error during model inference: {inference_err}")
            print(f"Input tensor shape during error: {X_tensor.shape}")
            traceback.print_exc()
            raise

        print(f"Individual trial predictions (raw): {all_predictions}")

        # Majority vote and metrics
        majority_prediction = int(np.bincount(all_predictions).argmax()) if len(all_predictions) > 0 else -1
        consistency_metrics = {"error": "Calculation failed or not applicable"}


        if num_trials > 0 and len(all_predictions) == num_trials:
            try:
                count = Counter(all_predictions)
                # Tie-break: default to 0
                if len(count) > 1 and len(count.most_common(2)) > 1 and count.most_common(2)[0][1] == count.most_common(2)[1][1]:
                    print("Warning: Tie in majority vote. Defaulting to 0 (Normal).")
                    majority_prediction = 0
                else:
                    majority_prediction = count.most_common(1)[0][0]

                print(f"Majority Prediction: {majority_prediction}")

                if num_trials > 1:
                    print("Calculating internal consistency metrics...")
                    y_true = np.full(num_trials, majority_prediction)
                    y_pred = all_predictions

                    accuracy = accuracy_score(y_true, y_pred)
                    # Determine number of classes robustly
                    if num_classes is None:
                        try:
                            num_classes = int(all_probabilities.shape[-1]) if hasattr(all_probabilities, 'shape') else int(np.max(y_pred)) + 1
                        except Exception:
                            num_classes = 2

                    if num_classes <= 2:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average='binary', pos_label=1, zero_division=0
                        )
                        _, specificity, _, _ = precision_recall_fscore_support(
                            y_true, y_pred, average='binary', pos_label=0, zero_division=0
                        )

                        unique_labels_in_preds = np.unique(y_pred)
                        if len(unique_labels_in_preds) == 1:
                            print(f"Warning: Only one class ({unique_labels_in_preds[0]}) predicted across trials.")
                            if unique_labels_in_preds[0] == 0:
                                tn = len(y_true) if majority_prediction == 0 else 0
                                fp = 0 if majority_prediction == 0 else len(y_true)
                                fn, tp = 0, 0
                            else:
                                tn, fp = 0, 0
                                fn = len(y_true) if majority_prediction == 0 else 0
                                tp = 0 if majority_prediction == 0 else len(y_true)
                            precision = 0 if tp + fp == 0 else tp / (tp + fp)
                            recall = 0 if tp + fn == 0 else tp / (tp + fn)
                            specificity = 0 if tn + fp == 0 else tn / (tn + fp)
                            f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
                        else:
                            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                            if cm.shape == (2, 2):
                                tn, fp, fn, tp = cm.ravel()
                            else:
                                print(f"Warning: Unexpected confusion matrix shape: {cm.shape}. Setting counts/metrics to 0.")
                                tn, fp, fn, tp = 0, 0, 0, 0
                                precision, recall, specificity, f1 = 0, 0, 0, 0

                        consistency_metrics = {
                            "num_trials": int(num_trials),
                            "num_normal_pred": int(np.sum(y_pred == 0)),
                            "num_alz_pred": int(np.sum(y_pred == 1)),
                            "accuracy": float(accuracy),
                            "precision": float(precision),
                            "recall_sensitivity": float(recall),
                            "specificity": float(specificity),
                            "f1_score": float(f1),
                            "true_positives": int(tp),
                            "true_negatives": int(tn),
                            "false_positives": int(fp),
                            "false_negatives": int(fn),
                            "majority_label_used_as_reference": int(majority_prediction)
                        }
                    elif num_classes == 3:
                        # For multi-class, get per-class metrics
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
                        )

                        # Specificity for each class: TN / (TN + FP)
                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                        tn_list, fp_list, fn_list, tp_list = [], [], [], []

                        for i in range(3):
                            tp = cm[i, i]
                            fn = cm[i, :].sum() - tp
                            fp = cm[:, i].sum() - tp
                            tn = cm.sum() - (tp + fn + fp)
                            tn_list.append(tn)
                            fp_list.append(fp)
                            fn_list.append(fn)
                            tp_list.append(tp)

                        specificity = [
                            0 if (tn_list[i] + fp_list[i]) == 0 else tn_list[i] / (tn_list[i] + fp_list[i])
                            for i in range(3)
                        ]

                        consistency_metrics = {
                            "num_trials": int(num_trials),
                            "num_class0_pred": int(np.sum(y_pred == 0)),
                            "num_class1_pred": int(np.sum(y_pred == 1)),
                            "num_class2_pred": int(np.sum(y_pred == 2)),
                            "accuracy": float(accuracy),
                            "precision_per_class": precision.tolist(),
                            "recall_per_class": recall.tolist(),
                            "specificity_per_class": specificity,
                            "f1_score_per_class": f1.tolist(),
                            "true_positives_per_class": tp_list,
                            "true_negatives_per_class": tn_list,
                            "false_positives_per_class": fp_list,
                            "false_negatives_per_class": fn_list
                        }
                    print(f"Consistency Metrics: {json.dumps(consistency_metrics, cls=NpEncoder, indent=4)}")
                else:
                    print("Only one trial/segment found, consistency metrics are not applicable.")
                    consistency_metrics = {"num_trials": 1, "message": "Metrics not applicable for single segment input"}

            except Exception as metrics_err:
                print(f"Error calculating metrics: {metrics_err}")
                traceback.print_exc()
                consistency_metrics = {"error": f"Metrics calculation failed: {metrics_err}"}

        else:
            print("Warning: No predictions available to calculate majority or metrics.")
            majority_prediction = -1
            consistency_metrics = {"error": "No predictions generated"}
            all_predictions = []

        # Prepare output
        # Derive class_names and label_mapping
        if num_classes is None and len(all_probabilities) > 0:
            try:
                num_classes = int(np.array(all_probabilities).shape[-1])
            except Exception:
                num_classes = 2
        if num_classes is None:
            num_classes = 2

        if num_classes == 3:
            class_names = ["cn", "ad", "mci"]
        elif num_classes == 2:
            class_names = ["normal", "alz"]
        else:
            class_names = [str(i) for i in range(num_classes)]
        label_mapping = {str(i): name for i, name in enumerate(class_names)}
        first_trial_probabilities = all_probabilities[0].tolist() if len(all_probabilities) > 0 else None

        results = {
            "majority_prediction": int(majority_prediction),
            "probabilities": first_trial_probabilities,
            "average_probabilities": (np.mean(all_probabilities, axis=0).tolist() if len(all_probabilities) > 0 else None),
            "trial_predictions": all_predictions.tolist() if isinstance(all_predictions, np.ndarray) else all_predictions,
            "consistency_metrics": consistency_metrics,
            "class_names": class_names,
            "label_mapping": label_mapping
        }

        # Save results to output.json
        output_file = 'output.json'
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, cls=NpEncoder, indent=4)
            print(f"Prediction results and metrics saved to {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"An error occurred saving results to {output_file}: {e}")
            traceback.print_exc()

        return results
