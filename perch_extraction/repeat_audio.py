from pydub import AudioSegment
import os


def repeat_audio_to_fill_duration(file_path, target_duration_ms):
    audio = AudioSegment.from_file(file_path, format="wav")
    duration_ms = len(audio)

    if duration_ms >= target_duration_ms:
        return audio[:target_duration_ms]  # Trim to target duration if longer

    repeated_audio = audio
    while len(repeated_audio) < target_duration_ms:
        repeated_audio += audio  # Repeat the audio until it exceeds the target duration

    return repeated_audio[:target_duration_ms]  # Trim to exact target duration


def process_folder(input_folder, output_folder, target_duration_ms=5000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output folder if it doesn't exist

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)

            # Repeat the audio clip to fill the 5-second duration
            repeated_audio = repeat_audio_to_fill_duration(file_path, target_duration_ms)

            # Save the processed audio clip
            repeated_audio.export(output_file_path, format='wav')


# folder paths
input_folder = 'AugmD19-files'
output_folder = 'AugmD19-looped'

# Process the folder
process_folder(input_folder, output_folder)
