# PoliPulse CA

**PoliPulse CA** is a Streamlit-based application for analysing transcribed political speeches using transformer models. It supports three main features:
- 📈 **General Trend Analysis** of sentiment or political tone
- 📊 **Group Comparison Analysis** between two sets of speeches
- 🏋️‍♂️ **Transformer Fine-Tuning** for political tone classification

## Features

- Upload `.txt`, `.docx`, or `.zip` speech files
- Select and configure transformer models for sentiment or tone analysis
- Apply statistical tests and visualise trends
- Fine-tune DistilBERT models with configurable training options

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/patrijo/PoliPulseCA.git
    cd PoliPulseCA
    ```

2. Install dependencies:
	```bash
    pip install -r requirements.txt
	```

3. Install the transformer models by running script:
	```bash
	python install_models.py		
	```
    <small>
- *Note: You can also install manually by downloading the two models and extracting them to their folders:
	place into sentiment_model: 
	distilbert_sentiment: https://drive.google.com/file/d/1YRs3IImXqkn3_iJg5HhrJbkhivy3L4fV/view?usp=drive_link
	place into politone_model:
	distilbert_politone: https://drive.google.com/file/d/18mlm08N71XfyrRhVOPyXGCoeAqyaLnZC/view?usp=drive_link*</small>

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Directory Structure

| Path                   | Description                                                 |
|------------------------|-------------------------------------------------------------|
| `app.py`               | Main Streamlit app to select and run analysis modes         |
| `mode_general.py`      | General sentiment/political tone trend analysis             |
| `mode_group_comparison.py` | Compare tone/sentiment across two speaker groups     |
| `mode_train_transformer.py` | Fine-tune transformer models via UI                |
| `shared_utils.py`      | Utility functions for preprocessing and scoring      |
| `sentiment_model/`     | Directory for local sentiment transformer models           |
| `politone_model/`      | Directory for political tone transformer models          |
| `offline_model/`      | Directory of offline transformer (base) for training         |
| `images/`              | UI assets like logos                                       |
| `trainingdata/`        | Excel data used for fine-tuning transformers                |
| `labelleddata/`        | Excel data with manually labelled and augmented political speech sentences            |
| `sous/`                | Transcriptions of State of the Union (SoU) addresses  |
| `testspeeches/`        | Transcriptions of testspeeches             |
| `README.md`            | This file                                                   |


## License

- **Code**: MIT License
- **Training data & docs**: CC BY 4.0
- **Transformer models**: Apache 2.0
- **SoU dataset/speech transcriptions**: Public Domain

## Author

- P. Jost
