# Odia-to-Hindi Translation Project

## Project Overview
This repository contains a prototype pipeline for Odia (Oriya) to Hindi translation. It includes data extraction, dataset preparation, morphological segmentation, vocabulary creation, a Transformer training workflow, and evaluation using BLEU, CHRF, and TER.

## Repository Contents
- `translation-or-to-hi.ipynb` — Main notebook for data preprocessing, model training, beam search decoding, and evaluation.
- `dataset_download.ipynb` — Notebook to download and filter Indic-align data from Hugging Face.
- `odia_hindi_extractor.ipynb` — Colab-oriented script to extract `ory_Orya` and `hin_Deva` pairs from raw CSV exports.
- `merge_odia_hindi.ipynb` — Notebook to merge multiple Odia-Hindi CSV files, clean duplicate entries, and save a unified corpus.
- `README.md` — Project documentation.

## Key Features
- Custom text cleaning for Odia and Hindi scripts
- Script validation for Odia and Devanagari characters
- Optional morphological segmentation using Morfessor
- Transformer-based encoder-decoder model implementation
- Training utilities with checkpointing, gradient accumulation, and mixed precision
- Beam search decoding for inference
- Evaluation with BLEU, CHRF, and TER metrics
- Data preparation and merge utilities for raw multilingual sources

## Dataset and Data Flow
- Default dataset path used in the main notebook is:
  `/kaggle/input/datasets/aditya83743/or-hi-corpus/merged_odia_hindi.csv`
- Expected column names for the translation corpus:
  - `ory_Orya` — Odia source text
  - `hin_Deva` — Hindi target text
- Data preparation includes:
  - cleaning HTML, URLs, email addresses, hashtags, and punctuation
  - filtering for valid Odia and Hindi script content
  - optional morphological segmentation for both source and target languages

## Requirements
The notebooks rely on the following Python packages:
- `sentencepiece`
- `morfessor`
- `sacrebleu`
- `sacremoses`
- `tqdm`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `torch`
- `numpy`
- `pandas`

> Note: `translation-or-to-hi.ipynb` installs most required packages automatically when its first cell runs.

## Usage
1. Open the notebooks in Jupyter Notebook, JupyterLab, Google Colab, or Kaggle.
2. Run the notebooks in the following recommended order:
   1. `dataset_download.ipynb` — download and filter the base dataset.
   2. `odia_hindi_extractor.ipynb` — extract Odia-Hindi pairs from raw CSV data.
   3. `merge_odia_hindi.ipynb` — merge multiple CSV sources and deduplicate.
   4. `translation-or-to-hi.ipynb` — preprocess data, train the model, and evaluate results.
3. If using a local dataset path, update `Config.DATASET_CSV` in `translation-or-to-hi.ipynb`.
4. Adjust hyperparameters in the `Config` class to tune vocabulary size, model depth, batch size, epochs, and decoding parameters.

## Configuration Notes
The main notebook includes a `Config` class with the following important fields:
- `DATASET_CSV` — path to the merged Odia-Hindi CSV file
- `OUTPUT_DIR`, `CKPT_DIR`, `MODEL_DIR`, `MORPH_DIR` — directories for outputs and checkpoints
- `SRC_VOCAB_SIZE`, `TGT_VOCAB_SIZE` — vocabulary sizes for Odia and Hindi
- `MAX_LEN`, `MIN_LEN` — sentence length filtering
- `USE_MORPH` — switch morphological segmentation on/off
- `BATCH_SIZE`, `MAX_EPOCHS`, `ACCUMULATE_GRAD` — training parameters
- `BEAM_SIZE`, `MAX_DECODE_LEN`, `LENGTH_PENALTY` — inference settings

## Recommendations
- Use GPU acceleration when training the Transformer model.
- If dataset loading fails, verify the `Config.DATASET_CSV` path and local file availability.
- If working locally instead of Kaggle, update output directories to accessible paths.
- Review sample translation output and metric trends in the notebook to validate model behavior.

## Deployment and Related Links
- Website: https://odia-to-hindi.vercel.app/
- Backend service (FastAPI inference endpoint): https://oa-biswajit-trans-or-hn-space.hf.space/docs#/
- Backend deployment repository (Hugging Face Space): https://huggingface.co/spaces/OA-Biswajit/trans_or_hn_space/tree/main
- Model and vocabulary (Hugging Face): https://huggingface.co/OA-Biswajit/or_hn_trans_models/tree/main
- Source code (GitHub): https://github.com/OA-Biswajit/Final_year_project

> Note: The backend is deployed on a Hugging Face Space, so the backend API endpoint and backend repository URL are related and may look similar.

## Contact
For questions about this workflow, inspect the notebook cells directly and modify the configuration to match your environment.
