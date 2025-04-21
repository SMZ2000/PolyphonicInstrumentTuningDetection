import os
import random
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import deeplake
from tqdm import tqdm
from huggingface_hub import create_repo, upload_folder
from multiprocessing import Pool, cpu_count

#CONFIG
OUTPUT_DIR = "out_of_tune_instruments"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
LABEL_CHUNK_PATH = os.path.join(OUTPUT_DIR, "labels_chunks")
REPO_ID = "mmthen/mixed_out_of_tune_instruments"  # 👈 CHANGE THIS

SAMPLE_RATE = 16000
DURATION_SEC = 4
MIX_SIZE = 3
TOTAL_SAMPLES = 300_000
PITCH_SHIFT_RANGE = (-1.5, 1.5)
CHUNK_SIZE = 10_000

INSTRUMENT_MAP = {
    0: 'string_bass', 1: 'bass_guitar', 18: 'flute', 20: 'trumpet', 21: 'trombone',
    22: 'tuba', 24: 'guitar_acoustic', 26: 'guitar_electric', 47: 'viola',
    48: 'violin', 50: 'saxophone', 54: 'oboe', 55: 'bassoon'
}
target_instruments = set(INSTRUMENT_MAP.keys())

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(LABEL_CHUNK_PATH, exist_ok=True)

#Load NSynth
ds = deeplake.load("hub://activeloop/nsynth-train", read_only=True)
filtered_samples = [
    (i, int(sample['instrument'].numpy()))
    for i, sample in tqdm(enumerate(ds), total=len(ds))
    if int(sample['instrument'].numpy()) in target_instruments
]

#Pitch Shift
def pitch_shift_audio(y, semitones, sr):
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return librosa.util.fix_length(y_shifted, size=len(y))

#Generate One Sample
def generate_sample(i):
    ds = deeplake.load("hub://activeloop/nsynth-train", read_only=True)

    chosen = random.sample(filtered_samples, MIX_SIZE)
    out_idx = random.randint(0, MIX_SIZE - 1)
    pitch_shift_amt = random.uniform(*PITCH_SHIFT_RANGE)

    mix = np.zeros(int(SAMPLE_RATE * DURATION_SEC), dtype=np.float32)
    label = {'filename': f"mix_{i}.wav", 'pitch_shift': round(pitch_shift_amt, 3)}
    instruments_all = []

    for idx, (sample_idx, inst_id) in enumerate(chosen):
        sample = ds[sample_idx]
        y = np.array(sample['audios'], dtype=np.float32).flatten()
        y = librosa.util.fix_length(y, size=int(SAMPLE_RATE * DURATION_SEC))

        inst_name = INSTRUMENT_MAP.get(inst_id, str(inst_id))
        instruments_all.append(inst_name)

        if idx == out_idx:
            y = pitch_shift_audio(y, pitch_shift_amt, SAMPLE_RATE)
            label['out_of_tune'] = inst_name
        else:
            label.setdefault('in_tune', []).append(inst_name)

        mix += y

    mix /= np.max(np.abs(mix) + 1e-6)
    file_path = os.path.join(AUDIO_DIR, label['filename'])
    sf.write(file_path, mix, SAMPLE_RATE)

    label['instruments_all'] = instruments_all
    return label

#Chunked Multiprocessing with Resume-Safe Logic
def run_batch(start_idx, end_idx, chunk_id):
    existing_files = set(os.listdir(AUDIO_DIR))
    indices_to_generate = [
        i for i in range(start_idx, end_idx)
        if f"mix_{i}.wav" not in existing_files
    ]

    if not indices_to_generate:
        print(f"Chunk {chunk_id} already complete. Skipping.")
        return

    print(f"Generating {len(indices_to_generate)} missing samples in chunk {chunk_id}...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(generate_sample, indices_to_generate), total=len(indices_to_generate)))

    pd.DataFrame(results).to_csv(os.path.join(LABEL_CHUNK_PATH, f"labels_chunk_{chunk_id}.csv"), index=False)

#MAIN
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print("Starting sample generation")
    chunk_id = 0
    for start in range(0, TOTAL_SAMPLES, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, TOTAL_SAMPLES)
        run_batch(start, end, chunk_id)
        chunk_id += 1

    #Combine CSV chunks
    print("Combining metadata")
    all_chunks = [pd.read_csv(os.path.join(LABEL_CHUNK_PATH, f)) for f in os.listdir(LABEL_CHUNK_PATH) if f.endswith(".csv")]
    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df.to_csv(os.path.join(OUTPUT_DIR, "labels.csv"), index=False)

    #Upload to Hugging Face
    print("Uploading to Hugging Face")
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    upload_folder(
        folder_path=OUTPUT_DIR,
        path_in_repo="",
        repo_id=REPO_ID,
        repo_type="dataset"
    )

    print("Dataset uploaded to:", REPO_ID)
