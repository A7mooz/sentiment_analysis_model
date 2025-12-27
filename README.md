# Sentiment Analysis Model

## Prerequisites
1. Python version 3.14 or higher
1. Initialize a venv environment (optional but recommended)
```bash
    python3 -m venv .venv
```
1. Install the required packages

```bash
    python3 -m pip install -r requirements.txt
```

## Preparation
1. Get a kraggle api key by going to your [user profile](https://www.kaggle.com/settings/account) and creating a new token then pasting it into the `.env` file under `KAGGLE_API_TOKEN` OR you could download the datasets from the links provided in `prepare.py` into the `datasets` folder then comment the kraggle download section.

1. Run `prepare.py`
```bash
    python3 src/prepare.py 
```

## Preprocessing
```bash
    python3 src/preprocess.py
```

## Training & Testing
```bash
    python3 src/train_model.py
```

## Run the api
```bash
    uvicorn src.main:app
```