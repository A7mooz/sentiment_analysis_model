import pandas as pd
import re
import nltk
import ftfy
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from joblib import Parallel, delayed

nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

preprocess_path = "datasets/processed/processed_data.csv"

#load
df: pd.DataFrame = pd.DataFrame(pd.read_csv("datasets/processed/prepared_data.csv", encoding="utf-8"), columns=["Comment","Sentiment"])

#switching the 0,1,2 to actual words
if df["Sentiment"].dtype != object:
    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    
    df["Sentiment"] = df["Sentiment"].map(lambda v: label_map[v])
stop_words = set(stopwords.words("english")) - {"not", "no", "never"}

#cleaning
def clean_text(text):
    txt = str(text)
    txt = ftfy.fix_text(txt)
    txt = txt.lower()
    txt = re.sub(r"http\S+|www.\S+", "", txt)
    txt = re.sub(r"[^a-z\s]", "", txt)
    words = word_tokenize(txt)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

#apply and save
print("Preprocessing...")

t1 = time.perf_counter()
df.dropna(inplace=True)

# df["Comment"] = df["Comment"].apply(clean_text) # takes forever
df["Comment"] = Parallel(backend="loky",n_jobs=-1)(delayed(clean_text)(text) for text in df["Comment"].to_numpy()) # Multithreading


df = df.loc[df["Comment"].str.split().str.len() > 2]
df["Comment"] = df["Comment"].str.strip()
df = df.loc[df["Comment"] != ""]
df = df.loc[df["Sentiment"].str.lower().isin(["positive", "neutral", "negative"])]
df['Sentiment'] = df["Sentiment"].str.lower()

df.dropna(inplace=True)

df.to_csv(preprocess_path, index=False)

t2 = time.perf_counter()

print(f"Saved proprocessed dataset to {preprocess_path} in {t2-t1:.2f}s")