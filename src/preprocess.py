# preprocess.py
import json, re, string
import pandas as pd
from nltk.corpus import stopwords

def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", "", txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def preprocess(input_file: str = "ENDEVR.json", output_file: str = "ENDEVR_clean.csv"):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except:
                    pass

    df = pd.DataFrame(data)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["clean"] = df["text"].apply(clean_text)

    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessed {len(df)} comments and saved to '{output_file}'")

if __name__ == "__main__":
    preprocess()
