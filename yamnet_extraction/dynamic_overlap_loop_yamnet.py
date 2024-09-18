from pydub import AudioSegment
import os
import math

def dynamic_loop_audio(file_path, target_frame_length_ms, overlap_percentage):
    """
    Dynamically loops the audio file based on the target frame length and overlap,
    ensuring a minimum length of 1 second (1000 ms) for each clip.

    Args:
    - file_path: Path to the input audio file.
    - target_frame_length_ms: Target length of each frame in milliseconds.
    - overlap_percentage: Percentage of overlap between frames.

    Returns:
    - A looped AudioSegment object.
    """
    audio = AudioSegment.from_file(file_path, format="wav")
    original_duration_ms = len(audio)

    # Calculate the step size based on the overlap
    step_size_ms = target_frame_length_ms * (1 - overlap_percentage)

    # Calculate the total required duration with the overlap
    total_required_duration_ms = math.ceil(original_duration_ms / step_size_ms) * step_size_ms

    # If the total required duration is less than the target frame length, set it to the target frame length
    if total_required_duration_ms < target_frame_length_ms:
        total_required_duration_ms = target_frame_length_ms

    # Ensure minimum length of 1 second
    if original_duration_ms < 960:
        total_required_duration_ms = max(total_required_duration_ms, 960)

    # Calculate how many times to loop the audio
    loop_count = math.ceil(total_required_duration_ms / original_duration_ms)

    # Loop the audio and trim to the required total duration
    looped_audio = (audio * loop_count)[:total_required_duration_ms]

    return looped_audio



def process_folder_dynamic_looping(input_folder, output_folder, frame_length_ms=960, overlap_percentage=0.5):
    """
    Processes each .wav file in the input folder, applying dynamic looping to ensure a minimum length of 1 second,
    and saves the result in the output folder.

    Args:
    - input_folder: Folder containing the input .wav files.
    - output_folder: Folder where the looped .wav files will be saved.
    - frame_length_ms: Length of each frame in milliseconds.
    - overlap_percentage: Overlap between frames as a percentage.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            looped_audio = dynamic_loop_audio(file_path, frame_length_ms, overlap_percentage)
            output_file_path = os.path.join(output_folder, file_name)
            looped_audio.export(output_file_path, format='wav')


input_folder = 'dynamic_overlap_input'
output_folder = 'dynamic_overlap_output'

# Process the folder with dynamic looping
process_folder_dynamic_looping(input_folder, output_folder)
