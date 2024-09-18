import time
import psutil
import logging
import tensorflow_hub as hub
import numpy as np
import csv
import io
import librosa
import re
import os
import pandas as pd

"""
file naming convention is
e.g. 18Y-117A-F1-04.wav - age-cat_id-gender(recordingCount)-meowCount
e.g. 5.25Y-099A-F1-NC-05.wav age-cat_id-gender(recordingCount)-context-meowCount
"""

# Setup basic configuration for logging
logging.basicConfig(filename='script_log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def log_resource_usage():
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory().percent
    logging.info(f"CPU Usage: {cpu}%, Memory Usage: {memory}%")


def extract_age_from_filename(filename):
    """Extracts the age part from the filename."""
    # Use regex to find the age pattern (e.g., "0.5Y" or "0Y")
    match = re.match(r"(\d+\.?\d*)Y", filename)
    if match:
        # Convert the matched age to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, return None or raise an error
        return None


def extract_pitch_from_filename(filename):
    """
    Extracts the pitch (mean F0) from the filename.

    The function assumes the pitch is at the end of the filename,
    preceded by a hyphen, and followed by the '.wav' extension.

    Parameters:
    - filename: The name of the file, as a string.

    Returns:
    - The extracted pitch as a float, or None if not found.
    """
    # Regex pattern to match a hyphen followed by one or more digits and potentially a decimal point
    pattern = re.compile(r'-(\d+\.\d+|\d+)\.wav$')
    match = pattern.search(filename)
    if match:
        # Convert the matched pitch to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, return None
        return None


# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


def extract_gender_from_filename(filename):
    """Extracts the gender part from the filename."""
    # Adjust regex to find the gender indicator more accurately
    match = re.search(r"([MFX])(?=-|\d|\.wav)", filename)
    if match:
        # Return the matched gender
        return match.group(1)
    else:
        # if gender not documented in filename return 'X' (UNKNOWN)
        return "X"


# Start tracking time and resources
start_time = time.time()
log_resource_usage()

# Directory containing the audio files
audio_dir = 'AugmD13-looped'

# List of audio files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

# Load the model.
model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4')

total_files = len(audio_files)
half_point = total_files // 2

processed_files = 0

data_list = []
for audio_file in audio_files:

    print(f"Processing {audio_file}")

    # Extract the filename from the full path
    filename = os.path.basename(audio_file)

    # Regex pattern to extract cat identifiers
    pattern = re.compile(r'-(\d{3}[A-Z])')
    # Search for the pattern in the filename
    match = pattern.search(filename)
    if match:
        # Extract cat identifier
        cat_id = match.group(1)
        print(cat_id)
    else:
        raise ValueError("Identifier missing or incorrect: ", filename)

    # Extract the target class from the filename
    age = extract_age_from_filename(filename)
    print(age)
    target_class = age

    # Extract the gender from the filename
    gender = extract_gender_from_filename(filename)
    print(gender)
    gender_class = gender

    # Full path to the audio file
    full_audio_path = os.path.join(audio_dir, audio_file)

    # Load the audio file using librosa
    waveform, sample_rate = librosa.load(full_audio_path, sr=32000, mono=True)

    # Scale and convert the waveform to the expected format.
    waveform = waveform.astype(np.float32)

    # Ensure the waveform is 5 seconds long at 32 kHz
    target_length = 5 * 32000  # 5 seconds * 32 kHz
    current_length = waveform.shape[0]

    # If the waveform is too long, trim it to the target length
    if current_length > target_length:
        waveform = waveform[:target_length]

    # If the waveform is too short, pad it with zeros to the target length
    elif current_length < target_length:
        padding = np.zeros(target_length - current_length, dtype=np.float32)
        waveform = np.concatenate((waveform, padding), axis=0)

    # Verify the waveform is the correct shape
    assert waveform.shape[0] == target_length, "The waveform is not the correct length."

    # Now you can pass the waveform to the model
    logits, embeddings = model.infer_tf(waveform[np.newaxis, :])

    # Extract pitch from filename (generated with crepe)
    mean_freq = extract_pitch_from_filename(filename)

    # Append the embeddings, mean frequency, target class, and cat identifier to the data list
    for embedding in embeddings.numpy():
        data_list.append([embedding, mean_freq, gender_class, target_class, cat_id])

    processed_files += 1
    if processed_files == half_point:
        # Log the time and resources at the halfway point
        elapsed_time = time.time() - start_time
        logging.info(f"Halfway Point: Total Time Elapsed: {elapsed_time:.2f} seconds")
        log_resource_usage()

# Create a DataFrame with embeddings and corresponding labels
embeddings_df = pd.DataFrame(data_list, columns=['embedding', 'mean_freq', 'gender', 'target', 'cat_id'])

# Expand 'embedding' column to separate columns
embeddings_df = pd.concat([pd.DataFrame(embeddings_df['embedding'].tolist()), embeddings_df['mean_freq'],
                           embeddings_df['gender'],
                           embeddings_df['target'],
                           embeddings_df['cat_id']], axis=1)

# Convert all column names to strings
embeddings_df.columns = embeddings_df.columns.astype(str)

# Save the DataFrame to a CSV file
embeddings_df.to_csv('perch_embeddings_test.csv', index=False)


elapsed_time = time.time() - start_time
logging.info(f"Total Time Elapsed: {elapsed_time:.2f} seconds")
log_resource_usage()

