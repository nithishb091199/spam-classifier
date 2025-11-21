import pandas as pd
import re
import os

RAW_DATA_PATH = "data/raw/SMSSpamCollection"
PROCESSED_DATA_PATH = "data/processed/cleaned.csv"


def clean_text(text):
    """
    Performs basic text cleaning:
    - lowercasing
    - removing URLs
    - removing non-alphanumeric characters
    - stripping extra spaces
    """

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # remove non-alphanumeric except spaces
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset():
    """Load raw dataset from file."""
    df = pd.read_csv(RAW_DATA_PATH, sep="\t", header=None, names=["label", "text"])
    return df


def preprocess(df):
    """Apply cleaning and label encoding."""
    # remove duplicates
    df = df.drop_duplicates()

    # remove missing values
    df = df.dropna()

    # clean text column
    df["clean_text"] = df["text"].apply(clean_text)

    # encode labels: ham = 0, spam = 1
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def save_processed(df):
    """Save processed dataset."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned dataset to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()

    print("Preprocessing dataset...")
    df_clean = preprocess(df)

    print(df_clean.head())

    print("\nSaving file...")
    save_processed(df_clean)

    print("\nDone!")
