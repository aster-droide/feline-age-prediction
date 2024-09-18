from pydub import AudioSegment
import os

def pad_audio_to_fixed_duration(file_path, target_duration_ms):
    audio = AudioSegment.from_file(file_path, format="wav")
    duration_ms = len(audio)
    # Pad the audio to reach the target duration
    padding_needed_ms = max(target_duration_ms - duration_ms, 0)  # Ensure non-negative
    padded_audio = audio + AudioSegment.silent(duration=padding_needed_ms)
    return padded_audio

def process_folder(folder_path, output_folder_path, target_duration_ms=5000):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            padded_audio = pad_audio_to_fixed_duration(file_path, target_duration_ms)
            output_file_path = os.path.join(output_folder_path, file_name)
            padded_audio.export(output_file_path, format='wav')

# folder paths
input_folder = 'input'
output_folder = 'output'

# Process the folder
process_folder(input_folder, output_folder)
