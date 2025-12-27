
# datasets
# https://www.kaggle.com/datasets/tariqsays/sentiment-dataset-with-1-million-tweets
# https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset
# https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset

import os

prepared_path = "datasets/processed/prepared_data.csv"

if not os.path.exists("datasets/processed"): os.makedirs("datasets/processed")
with open("datasets/.gitignore", "w") as f:
    f.write("*")

if not os.path.exists("out"): os.makedirs("out")
with open("out/.gitignore", "w") as f:
    f.write("*")

# Downloading datasets from kaggle

from dotenv import load_dotenv

load_dotenv()

os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN", "")

import kaggle

kaggle.api.dataset_download_files("tariqsays/sentiment-dataset-with-1-million-tweets", path="datasets", unzip=True)
kaggle.api.dataset_download_files("abdelmalekeladjelet/sentiment-analysis-dataset", path="datasets", unzip=True)
kaggle.api.dataset_download_files("abhi8923shriv/sentiment-analysis-dataset", path="datasets", unzip=True)

# end download you could comment this

import pandas as pd

# Data that we save into
data = {'Comment': [], 'Sentiment': []}

# Parse tweets dataset
df = pd.read_csv("datasets/dataset.csv")
df = df[df["Language"] == "en"]

data['Comment'] += df["Text"].tolist()
data['Sentiment'] += df["Label"].str.lower().tolist()

# Parse 2nd dataset
df = pd.read_csv("datasets/sentiment_data.csv")
df = df[["Comment", "Sentiment"]]

# Map numbers to string labels
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
df["Sentiment"] = df["Sentiment"].map(label_map)

data['Comment'] += df["Comment"].tolist()
data['Sentiment'] += df["Sentiment"].str.lower().tolist()

# Parse 3rd dataset
df = pd.read_csv("datasets/train.csv", encoding="ISO-8859-1")
df = df[["text", "sentiment"]]
data['Comment'] += df["text"].tolist()
data['Sentiment'] += df["sentiment"].str.lower().tolist()

df = pd.DataFrame(data)

df.to_csv(prepared_path, index=False)

if not os.path.exists("out"): os.mkdir("out")
with open("datasets/.gitignore", "w") as f:
    f.write("*")