import json
import numpy as np

# Custom JSON encoder for numpy data types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)  # Convert numpy integers to Python int
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Represent NaN/Inf as null in JSON
            return float(obj)  # Convert numpy floats to Python float
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        if isinstance(obj, (np.bool_,)):
            return bool(obj)  # Convert numpy bool to Python bool
        return super(NpEncoder, self).default(obj)

# Replace non-ASCII characters for Helvetica compatibility
def sanitize_for_helvetica(text_input):
    if not isinstance(text_input, str):
        text_input = str(text_input)
    # Map of non-ASCII to ASCII replacements
    replacements = {
        '•': '-',
        '◦': '-',
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '-',
        '…': '...',
        '€': 'EUR',
        '£': 'GBP',
    }
    for uni_char, ascii_char in replacements.items():
        text_input = text_input.replace(uni_char, ascii_char)
    # Replace remaining non-ASCII with '?'
    return "".join(c if ord(c) < 128 else "?" for c in text_input)
