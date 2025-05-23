# ===================================================================================================================
# install_models.py
# Download an installs the distilbert sentiment and fine-tuned politone transformer models 
# from their google drive storage and extracts them to the required folders 
# Author: P. Jost
# Date: May 2025
# License: MIT License (code), CC BY 4.0 (training data/tables/docs) https://creativecommons.org/licenses/by/4.0/
#          Transformer models: Apache 2.0; data: US public domain/open
# ===================================================================================================================

# ───────────────────────────────────────────────────────────────────────────────────────────
# PoliPulse CA - GENERAL SENTIMENT/POLITONE TREND ANALYSIS & VISUALISATION
# ───────────────────────────────────────────────────────────────────────────────────────────


import os
import gdown
import zipfile

# List of models to download and extract
MODELS = [
    {
        "name": "Offline DistilBERT",
        "folder": "offline-models",
        "file_id": "1YRs3IImXqkn3_iJg5HhrJbkhivy3L4fV",
        "filename": "model1.zip",
    },
    {
        "name": "PoliTone Model",
        "folder": "politone_model",
        "file_id": "18mlm08N71XfyrRhVOPyXGCoeAqyaLnZC",
        "filename": "model2.zip",
    },
    {
        "name": "Sentiment Model",
        "folder": "sentiment_model",
        "file_id": "1YRs3IImXqkn3_iJg5HhrJbkhivy3L4fV",
        "filename": "model3.zip",
    },
]

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_and_extract(model):
    folder = model["folder"]
    file_id = model["file_id"]
    zip_filename = model["filename"]
    zip_path = os.path.join(folder, zip_filename)

    print(f"\n📦 Downloading {model['name']}...")
    ensure_folder(folder)

    if os.path.exists(zip_path):
        print(f"✔️ {zip_filename} already exists, skipping download.")
    else:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, zip_path, quiet=False)

    # Extract the zip file
    print(f"📂 Extracting {zip_filename} to {folder}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)
        print(f"✅ Extraction complete.")
        os.remove(zip_path)
        print(f"🗑️ Removed zip file: {zip_filename}")
    except zipfile.BadZipFile:
        print(f"❌ Error: {zip_filename} is not a valid zip file.")

def install_all():
    for model in MODELS:
        download_and_extract(model)

if __name__ == "__main__":
    install_all()