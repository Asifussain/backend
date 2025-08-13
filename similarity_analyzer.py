import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.stats import zscore
import os
import math
import io
import base64
import traceback

# Default reference files and channel index
DEFAULT_ALZHEIMER_REF_FILE = 'feature_07.npy'
DEFAULT_NORMAL_REF_FILE = 'feature_35.npy'
DEFAULT_CHANNEL_INDEX_TO_PLOT = 0

def configure_plot_style():
    """Set Matplotlib style for plots."""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.facecolor'] = '#222b3b'
        plt.rcParams['axes.facecolor'] = '#222b3b'
        plt.rcParams['grid.color'] = '#444444'
        plt.rcParams['grid.linestyle'] = ':'
        plt.rcParams['text.color'] = '#DDDDDD'
        plt.rcParams['axes.labelcolor'] = '#CCCCCC'
        plt.rcParams['xtick.color'] = '#CCCCCC'
        plt.rcParams['ytick.color'] = '#CCCCCC'
        plt.rcParams['axes.edgecolor'] = '#555555'
        plt.rcParams['legend.facecolor'] = '#333f50'
        plt.rcParams['legend.edgecolor'] = '#555555'
        plt.rcParams['legend.framealpha'] = 0.8
        plt.rcParams['savefig.facecolor'] = '#222b3b'
        plt.rcParams['savefig.edgecolor'] = '#222b3b'
    except Exception as e:
        print(f"Warning: Could not apply seaborn style - {e}. Using default.")

def load_and_prepare_eeg(file_path, file_desc):
    """Load EEG data or create dummy if missing. Ensure shape (samples, channels)."""
    if not os.path.exists(file_path):
        print(f"Warning: {file_desc} file not found at {file_path}. Creating dummy data.")
        try:
            num_samples = 5 * 128
            num_channels = 19
            time = np.arange(num_samples) / 128.0
            noise_level = 2
            base_freq = 10

            if 'Normal' in file_desc:
                freq1 = base_freq + np.random.uniform(-1, 1)
                amp1 = 5
            elif 'Alzheimer' in file_desc:
                freq1 = base_freq + np.random.uniform(-3, -1)
                amp1 = 4
            else:
                freq1 = base_freq + np.random.uniform(-2, 2)
                amp1 = 4.5

            dummy_data = np.zeros((num_samples, num_channels))
            for ch in range(num_channels):
                noise = np.random.randn(num_samples) * noise_level
                dummy_data[:, ch] = amp1 * np.sin(2 * np.pi * freq1 * time + np.random.rand()*np.pi) + noise
            print(f"Generated dummy {file_desc} data with shape {dummy_data.shape}")
            return dummy_data.astype(np.double)
        except Exception as e:
            print(f"Error creating dummy data for {file_desc}: {e}")
            return None

    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded {file_desc} data. Original shape: {data.shape}")

        if data.ndim == 3:
            eeg_data = data[0, :, :]
            print(f"Using first trial from {file_desc}. Shape: {eeg_data.shape}")
        elif data.ndim == 2:
            eeg_data = data
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}. Expected 2 or 3.")

        if eeg_data.shape[0] < eeg_data.shape[1]:
            print(f"Warning: {file_desc} data seems (channels, samples). Transposing.")
            eeg_data = eeg_data.T

        if eeg_data.shape[1] != 19:
            print(f"Warning: {file_desc} data has {eeg_data.shape[1]} channels, expected 19.")

        print(f"{file_desc} data shape for analysis: {eeg_data.shape}")
        return eeg_data.astype(np.double)

    except Exception as e:
        print(f"Error loading/processing {file_desc} file {file_path}: {e}")
        traceback.print_exc()
        return None

def calculate_channel_similarities(sample_data, norm_ref_data, alz_ref_data):
    """Compute DTW distances for each channel."""
    if sample_data is None or norm_ref_data is None or alz_ref_data is None:
        print("Error: One or more input datasets are None in calculate_channel_similarities.")
        return []
    if not (sample_data.shape == norm_ref_data.shape == alz_ref_data.shape):
        print(f"Error: Data shapes mismatch in calculate_channel_similarities: Sample={sample_data.shape}, Normal={norm_ref_data.shape}, Alz={alz_ref_data.shape}")
        return []

    n_samples, n_channels = sample_data.shape
    results = []

    print(f"Calculating DTW similarities for {n_channels} channels...")
    for i in range(n_channels):
        sample_ch = sample_data[:, i]
        norm_ref_ch = norm_ref_data[:, i]
        alz_ref_ch = alz_ref_data[:, i]

        # Z-score normalization
        sample_std = np.std(sample_ch)
        norm_ref_std = np.std(norm_ref_ch)
        alz_ref_std = np.std(alz_ref_ch)

        sample_ch_norm = zscore(sample_ch) if sample_std > 1e-9 else sample_ch - np.mean(sample_ch)
        norm_ref_ch_norm = zscore(norm_ref_ch) if norm_ref_std > 1e-9 else norm_ref_ch - np.mean(norm_ref_ch)
        alz_ref_ch_norm = zscore(alz_ref_ch) if alz_ref_std > 1e-9 else alz_ref_ch - np.mean(alz_ref_ch)

        dist_norm = np.inf
        dist_alz = np.inf
        closer_to = "Error"

        try:
            dist_norm = dtw.distance_fast(sample_ch_norm.astype(np.double), norm_ref_ch_norm.astype(np.double), use_pruning=True)
            dist_alz = dtw.distance_fast(sample_ch_norm.astype(np.double), alz_ref_ch_norm.astype(np.double), use_pruning=True)
            closer_to = "Normal" if dist_norm <= dist_alz else "Alzheimer"
        except ImportError:
            print(f"Info: C implementation of DTW not found. Using slower Python version for channel {i+1}.")
            try:
                dist_norm = dtw.distance(sample_ch_norm, norm_ref_ch_norm)
                dist_alz = dtw.distance(sample_ch_norm, alz_ref_ch_norm)
                closer_to = "Normal" if dist_norm <= dist_alz else "Alzheimer"
            except Exception as e_std:
                print(f"Error: Standard DTW failed for channel {i+1}: {e_std}")
        except Exception as e_fast:
            print(f"Warning: dtw.distance_fast failed for channel {i+1} ({e_fast}). Trying standard DTW.")
            try:
                dist_norm = dtw.distance(sample_ch_norm, norm_ref_ch_norm)
                dist_alz = dtw.distance(sample_ch_norm, alz_ref_ch_norm)
                closer_to = "Normal" if dist_norm <= dist_alz else "Alzheimer"
            except Exception as e_std:
                print(f"Error: Standard DTW also failed for channel {i+1}: {e_std}")

        results.append({
            'channel': i,
            'dist_norm': float(dist_norm) if np.isfinite(dist_norm) else None,
            'dist_alz': float(dist_alz) if np.isfinite(dist_alz) else None,
            'closer_to': closer_to
        })

        if (i + 1) % 5 == 0 or i == n_channels - 1:
            print(f"  Processed channel {i+1}/{n_channels}...")

    print("Finished DTW calculations.")
    return results

def plot_single_channel_comparison(sample_ch, norm_ref_ch, alz_ref_ch, channel_index):
    """Plot one channel's normalized signals for comparison."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    try:
        n_samples = len(sample_ch)
        time = np.arange(n_samples)

        s_norm = zscore(sample_ch) if np.std(sample_ch) > 1e-9 else sample_ch - np.mean(sample_ch)
        n_norm = zscore(norm_ref_ch) if np.std(norm_ref_ch) > 1e-9 else norm_ref_ch - np.mean(norm_ref_ch)
        a_norm = zscore(alz_ref_ch) if np.std(alz_ref_ch) > 1e-9 else alz_ref_ch - np.mean(alz_ref_ch)

        ax.plot(time, s_norm, color='#FFFFFF', lw=1.8, label='Your Sample (Normalized)', zorder=3)
        ax.plot(time, n_norm, color='#3498db', lw=1.5, linestyle='--', alpha=0.85, label='Normal Ref. (Normalized)', zorder=2)
        ax.plot(time, a_norm, color='#e74c3c', lw=1.5, linestyle=':', alpha=0.85, label="Alzheimer's Ref. (Normalized)", zorder=1)

        ax.set_xlabel("Time (samples)", fontsize=11)
        ax.set_ylabel("Signal Shape (Z-score)", fontsize=11)
        ax.set_title(f"Channel {channel_index + 1}: Signal Shape Comparison", fontsize=14, weight='bold', color='#EEEEEE')
        ax.legend(loc='upper right', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.margins(x=0.02)
        plt.tight_layout()
    except Exception as plot_err:
        print(f"Error during plotting channel {channel_index + 1}: {plot_err}")
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error plotting channel {channel_index + 1}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, color='red', fontsize=12, wrap=True)
    return fig

def fig_to_base64(fig):
    """Convert Matplotlib figure to base64 PNG string."""
    if fig is None: return None
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error converting figure to base64: {e}")
        return None
    finally:
        plt.close(fig)

def run_similarity_analysis(sample_file_path, alz_ref_file_path=None, norm_ref_file_path=None, channel_to_plot=DEFAULT_CHANNEL_INDEX_TO_PLOT):
    """Main analysis: load data, compute similarities, plot, and return results."""
    print("--- Starting EEG Similarity Analysis ---")

    script_dir = os.path.dirname(__file__)
    if alz_ref_file_path is None:
        alz_ref_file_path = os.path.join(script_dir, DEFAULT_ALZHEIMER_REF_FILE)
    if norm_ref_file_path is None:
        norm_ref_file_path = os.path.join(script_dir, DEFAULT_NORMAL_REF_FILE)

    sample_data = load_and_prepare_eeg(sample_file_path, "User Sample")
    alz_ref_data = load_and_prepare_eeg(alz_ref_file_path, "Alzheimer's Reference")
    norm_ref_data = load_and_prepare_eeg(norm_ref_file_path, "Normal Reference")

    analysis_results = {
        'error': None,
        'channel_results': [],
        'normal_closer_count': 0,
        'alz_closer_count': 0,
        'error_channels_count': 0,
        'total_channels': 19,
        'overall_similarity': 'Error',
        'interpretation': '',
        'plot_base64': None,
        'plotted_channel_index': None
    }

    if sample_data is None:
        error_msg = f"Error: Could not load/generate User Sample data from {sample_file_path}."
        analysis_results.update({'error': error_msg, 'interpretation': error_msg})
        print(error_msg)
        return analysis_results
    if alz_ref_data is None:
        error_msg = f"Error: Could not load/generate Alzheimer's Ref data from {alz_ref_file_path}."
        analysis_results.update({'error': error_msg, 'interpretation': error_msg})
        print(error_msg)
        return analysis_results
    if norm_ref_data is None:
        error_msg = f"Error: Could not load/generate Normal Ref data from {norm_ref_file_path}."
        analysis_results.update({'error': error_msg, 'interpretation': error_msg})
        print(error_msg)
        return analysis_results

    n_samples, n_channels = sample_data.shape
    analysis_results['total_channels'] = n_channels

    if not (sample_data.shape == alz_ref_data.shape == norm_ref_data.shape):
        error_msg = f"Error: Data shapes incompatible. Sample: {sample_data.shape}, Alz Ref: {alz_ref_data.shape}, Norm Ref: {norm_ref_data.shape}"
        analysis_results.update({'error': error_msg, 'interpretation': error_msg})
        print(error_msg)
        return analysis_results

    print(f"\nData loaded. Comparing {n_channels} channels, {n_samples} samples each.")

    if not isinstance(channel_to_plot, int) or not (0 <= channel_to_plot < n_channels):
        print(f"Warning: Invalid channel index '{channel_to_plot}'. Defaulting to channel {DEFAULT_CHANNEL_INDEX_TO_PLOT + 1}.")
        channel_to_plot = DEFAULT_CHANNEL_INDEX_TO_PLOT
    analysis_results['plotted_channel_index'] = channel_to_plot

    try:
        print("\n--- Calculating Similarity (DTW Distance) ---")
        channel_results = calculate_channel_similarities(sample_data, norm_ref_data, alz_ref_data)
        analysis_results['channel_results'] = channel_results

        valid_results = [r for r in channel_results if r.get('closer_to') != "Error"]
        norm_closer = sum(1 for r in valid_results if r.get('closer_to') == 'Normal')
        alz_closer = sum(1 for r in valid_results if r.get('closer_to') == 'Alzheimer')
        error_channels = n_channels - len(valid_results)

        analysis_results['normal_closer_count'] = norm_closer
        analysis_results['alz_closer_count'] = alz_closer
        analysis_results['error_channels_count'] = error_channels

        if error_channels == n_channels:
            analysis_results['overall_similarity'] = "Error during comparison"
        elif norm_closer > alz_closer:
            analysis_results['overall_similarity'] = "Higher Similarity to Normal Pattern"
        elif alz_closer > norm_closer:
            analysis_results['overall_similarity'] = "Higher Similarity to Alzheimer's Pattern"
        else:
            analysis_results['overall_similarity'] = "Inconclusive / Equidistant Similarity"

        interp = f"Similarity Analysis (DTW):\n"
        interp += f"- Overall Assessment: {analysis_results['overall_similarity']}\n"
        if n_channels > 0:
            interp += f"- Channels More Similar to Normal Ref: {norm_closer} ({norm_closer/n_channels*100:.1f}%)\n"
            interp += f"- Channels More Similar to Alzheimer's Ref: {alz_closer} ({alz_closer/n_channels*100:.1f}%)\n"
            if error_channels > 0:
                interp += f"- Channels with Errors during DTW: {error_channels}\n"
        else:
            interp += "- No channels found for comparison.\n"
        interp += "\nDisclaimer: This analysis compares overall signal shapes using Dynamic Time Warping (DTW) against reference patterns and does not constitute a medical diagnosis. Results indicate pattern similarity, not disease presence. Consult a healthcare professional for diagnosis."
        analysis_results['interpretation'] = interp

        print("\n--- Similarity Summary ---")
        print(f"Overall Assessment: {analysis_results['overall_similarity']}")
        print(f"Channels closer to Normal: {norm_closer}/{n_channels}")
        print(f"Channels closer to Alzheimer's: {alz_closer}/{n_channels}")
        if error_channels > 0: print(f"Channels with DTW errors: {error_channels}/{n_channels}")

        print(f"\n--- Generating Plot for Channel {channel_to_plot + 1} ---")
        fig_channel = plot_single_channel_comparison(
            sample_data[:, channel_to_plot],
            norm_ref_data[:, channel_to_plot],
            alz_ref_data[:, channel_to_plot],
            channel_to_plot
        )
        analysis_results['plot_base64'] = fig_to_base64(fig_channel)
        if analysis_results['plot_base64']:
            print(f" -> Plot for Channel {channel_to_plot + 1} generated.")
        else:
            print(f" -> Plot generation failed for Channel {channel_to_plot + 1}.")
            analysis_results['error'] = analysis_results['error'] or "Plot generation failed"

    except Exception as e:
        error_msg = f"An error occurred during similarity analysis: {e}"
        print(f"\n{error_msg}")
        traceback.print_exc()
        analysis_results['error'] = error_msg
        analysis_results['interpretation'] = f"Similarity analysis failed: {e}"

    print("\n--- Similarity Analysis Complete ---")
    return analysis_results

if __name__ == "__main__":
    SAMPLE_FILE_FOR_TEST = 'sample.npy'
    ALZ_REF_FILE_FOR_TEST = DEFAULT_ALZHEIMER_REF_FILE
    NORM_REF_FILE_FOR_TEST = DEFAULT_NORMAL_REF_FILE
    CHANNEL_TO_PLOT_FOR_TEST = DEFAULT_CHANNEL_INDEX_TO_PLOT

    print(f"\n--- Running Standalone Test ---")
    print(f"Sample: {SAMPLE_FILE_FOR_TEST}")
    print(f"Alz Ref: {ALZ_REF_FILE_FOR_TEST}")
    print(f"Norm Ref: {NORM_REF_FILE_FOR_TEST}")
    print(f"Plot Channel: {CHANNEL_TO_PLOT_FOR_TEST + 1}")

    results = run_similarity_analysis(
        sample_file_path=SAMPLE_FILE_FOR_TEST,
        alz_ref_file_path=ALZ_REF_FILE_FOR_TEST,
        norm_ref_file_path=NORM_REF_FILE_FOR_TEST,
        channel_to_plot=CHANNEL_TO_PLOT_FOR_TEST
    )

    print("\n--- Final Results Dictionary (Summary for Test Run) ---")
    printable_results = {k: v for k, v in results.items() if k not in ['plot_base64', 'channel_results']}
    import json
    print(json.dumps(printable_results, indent=2))

    if results.get('plot_base64') and not results.get('error'):
        print("\nPlot generated (base64 data omitted from console output).")
    elif results.get('error'):
        print(f"\nAnalysis finished with error: {results['error']}")
    else:
        print("\nPlot was not generated.")

    print("\n--- Standalone Test Complete ---")
