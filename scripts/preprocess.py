import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm

# -------------------------------------------------
# Resolve project root dynamically
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Find all .rec files recursively
# -------------------------------------------------
rec_files = sorted([f for f in RAW_DIR.rglob("*.rec")])
print(f"[INFO] Found {len(rec_files)} .rec files")

# -------------------------------------------------
# Sleep stage mapping (standardized)
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

def parse_hypnogram(hyp_file):
    with open(hyp_file, "rb") as f:
        content = f.read()
    # Keep only ASCII printable characters
    text = "".join(chr(b) for b in content if 32 <= b <= 126 or b in (10, 13))
    # Extract numbers at the end (the real hypnogram)
    import re
    numbers = [int(x) for x in re.findall(r'\b[0-5]\b', text)]
    # Map 4 → 4, 3 → 3, etc., keep -1 for unknown (if needed)
    SLEEP_STAGE_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:-1}  # 5=movement
    labels = [SLEEP_STAGE_MAP.get(n, -1) for n in numbers]
    return np.array(labels)

def preprocess_file(rec_file):
    hyp_file = rec_file.with_suffix(".hyp")
    if not hyp_file.exists():
        print(f"[WARN] Missing hypnogram for {rec_file.name}")
        return

    # ---- TEMP RENAME FIX FOR MNE ----
    temp_edf = rec_file.with_suffix(".edf")
    rec_file.rename(temp_edf)

    try:
        raw = mne.io.read_raw_edf(temp_edf, preload=True, verbose=False)

        # Pick EEG channels only
        eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
        if not eeg_channels:
            print(f"[WARN] No EEG channels found in {rec_file.name}")
            return
        raw.pick_channels(eeg_channels)

        data = raw.get_data()
        labels = parse_hypnogram(hyp_file)

        # Check for length mismatch
        if data.shape[1] != len(labels):
            min_len = min(data.shape[1], len(labels))
            data = data[:, :min_len]
            labels = labels[:min_len]

        base = rec_file.stem
        np.save(PROCESSED_DIR / f"{base}_X.npy", data)
        np.save(PROCESSED_DIR / f"{base}_y.npy", labels)

    finally:
        # Rename back no matter what
        temp_edf.rename(rec_file)

# -------------------------------------------------
# Run preprocessing
# -------------------------------------------------
for rec in tqdm(rec_files, desc="Preprocessing Sleep-EDF"):
    preprocess_file(rec)

print("[SUCCESS] Preprocessing completed correctly.")
print(f"[INFO] Saved to: {PROCESSED_DIR}")
