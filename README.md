# This work is licensed under CC BY, please cite [this paper](https://rdcu.be/eJi5P) when using this work: 

van Toor, A., Qazi, N. & Paladini, S. A deep learning pipeline for age prediction from vocalisations of the domestic feline. _Sci Rep_ **15**, 34565 (2025). https://doi.org/10.1038/s41598-025-17986-z

# feline-age-prediction
Dev scripts, analysis, and model builds for a feline age prediction machine learning pipeline 

## Dataset

The [dataset]([url](https://github.com/aster-droide/feline-age-prediction/tree/main/dataset)) folder contains the audio files that have been cropped out and labelled from extended recordings. The [raw_audio]([url](https://github.com/aster-droide/feline-age-prediction/tree/main/dataset/raw_audio)) folder contains audio files looped to the respective embeddings windows of Perch, VGGish, and YAMNet. VGGish recordings are looped slightly longer than YAMNet as VGGish internally crops to the last embeddings window, YAMNet is looped to just below the last embedding window as any recordings on or over would automatically be silence padded - this is to leave a buffer. The [embeddings]([url](https://github.com/aster-droide/feline-age-prediction/tree/main/dataset/embeddings)) folder contains the respective extracted embeddings.  

## Feature extraction

Feature extraction from selected audio transfer learning (TL) models can be found in [perch_extraction]([url](https://github.com/aster-droide/feline-age-prediction/tree/main/perch_extraction)), [yamnet_extraction]([url](https://github.com/aster-droide/feline-age-prediction/tree/main)), and in a separate repository for [VGGish]([url](https://github.com/aster-droide/audioset-thesis-work/blob/main/audioset/vggish/vggish_inference_demo.py)).

## Models, code, and analysis

[models_analysis_notebooks]([url](https://github.com/aster-droide/feline-age-prediction/tree/main/models_analysis_notebooks)) contains notebooks for each TL model including binary and categorical analysis, models trained and assessed over five different seeds across four folds for each seed. Analysis is provided for each seed and final results are visible at the bottom of the notebook. Optuna tuning notebook scripts included. 

## Contact Information

For further inquiries or collaboration, please contact:

Astrid van Toor - astridvantoor@gmail.com / avtoor@blueoasis.pt
