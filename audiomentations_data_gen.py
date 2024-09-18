import librosa
import os
import audiomentations as am
import soundfile as sf

"""
file naming convention is
e.g. 18Y-117A-F1-04.wav - age-cat_id-gender(recordingCount)-meowCount
e.g. 5.25Y-099A-F1-NC-05.wav age-cat_id-gender(recordingCount)-context-meowCount
"""

def create_augmenter(cat_class):
    if float(cat_class) < 0.5:  # Kitten
        return am.Compose([
            am.PitchShift(min_semitones=-1, max_semitones=1, p=0.7),
            am.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.7),
            am.Gain(min_gain_in_db=-5, max_gain_in_db=5, p=0.6)
        ])
    elif 0.5 <= float(cat_class) < 10:  # Adult
        return am.Compose([
            am.PitchShift(min_semitones=-1, max_semitones=0.5, p=0.7),
            am.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.7),
            am.Gain(min_gain_in_db=-5, max_gain_in_db=5, p=0.6)
        ])
    else:  # Senior
        return am.Compose([
            am.PitchShift(min_semitones=-1.2, max_semitones=-0.4, p=0.5),
            am.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.7),
            am.Gain(min_gain_in_db=-5, max_gain_in_db=5, p=0.6)
        ])

def augment_audio(file_path_param, output_dir_param, cat_class, num_augmented=5, original_sr=16000):
    """
    Augment an audio file by applying various transformations: https://iver56.github.io/audiomentations/
    Save the augmented versions in the specified output directory with 'aug_' prefix.

    file naming convention is
    e.g. 18Y-117A-F1-04.wav - age-cat_id-gender(recordingCount)-meowCount
    """
    # Load the file with the specified original sample rate to avoid automatic resampling
    y, sr = librosa.load(file_path_param, sr=original_sr)
    file_name = os.path.basename(file_path_param)

    augmenter = create_augmenter(cat_class)

    for i in range(num_augmented):
        y_augmented = augmenter(y, sr)
        aug_file_name = file_name.replace('.wav', f'_aug_{i}.wav')
        aug_path = os.path.join(output_dir_param, aug_file_name)
        sf.write(aug_path, y_augmented, sr)

# Apply augmentation to each file in the dataset directory
dataset_dir = 'Everything'
output_dir = 'augmented-publication-work/D13'

for fn in os.listdir(dataset_dir):
    if fn.endswith('.wav'):
        file_path = os.path.join(dataset_dir, fn)

        cat_class = fn.split('-')[0][:-1]  # remove 'Y' at end

        if 0.5 <= float(cat_class) < 10:
            num_aug = 2
        elif float(cat_class) < 0.5:
            num_aug = 1 # less augmentation for majority class
        else:  # senior
            num_aug = 4

        augment_audio(file_path, output_dir, cat_class, num_augmented=num_aug)
