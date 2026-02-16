import numpy as np
from pathlib import Path

# -------------------------------------------------
# Resolve project root dynamically
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# -------------------------------------------------
# Sleep stage mapping (consistent with preprocessing)
# -------------------------------------------------
SLEEP_STAGE_MAP = {
    "W": 0,   # Wake
    "1": 1,   # N1
    "2": 2,   # N2
    "3": 3,   # N3
    "4": 3,   # N4 → merged into N3
    "R": 4,   # REM
    "M": 5,   # Movement
    "?": -1   # Unknown
}

def load_sleepedf():
    """
    Load all processed Sleep-EDF .npy files and concatenate them.
    Returns:
        X: np.ndarray of shape (num_channels, total_samples)
        y: np.ndarray of shape (total_samples,)
    """
    rec_X_files = sorted(PROCESSED_DIR.glob("*_X.npy"))
    rec_y_files = sorted(PROCESSED_DIR.glob("*_y.npy"))

    if not rec_X_files or not rec_y_files:
        raise RuntimeError(f"No processed data found in {PROCESSED_DIR}")

    X_list, y_list = [], []

    for x_file, y_file in zip(rec_X_files, rec_y_files):
        x = np.load(x_file)
        y = np.load(y_file)

        # Ensure labels length matches data length along time axis
        if x.shape[1] != y.shape[0]:
            print(f"[WARNING] Length mismatch in {x_file.stem}, skipping.")
            continue

        X_list.append(x)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No valid data-label pairs loaded.")

    # Concatenate along time axis
    X_all = np.concatenate(X_list, axis=1)  # channels x total_time
    y_all = np.concatenate(y_list, axis=0)  # total_time

    print(f"[INFO] Final dataset shape: X={X_all.shape}, y={y_all.shape}")
    print(f"[INFO] Unique classes: {set(y_all)}")

    return X_all, y_all

# -------------------------------------------------
# Standalone test
# -------------------------------------------------
if __name__ == "__main__":
    X, y = load_sleepedf()
