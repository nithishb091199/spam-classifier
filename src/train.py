import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

DATA_PATH = "data/processed/cleaned.csv"
MODEL_PATH = "data/processed/spam_model.joblib"


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def create_pipeline():
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_df=0.9,
                ),
            ),
            ("clf", MultinomialNB()),
        ]
    )
    return pipeline


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    print("Loading cleaned data...")
    df = load_data()

    # ---- Defensive cleaning: ensure no NaNs / empty texts ----
    # If 'clean_text' doesn't exist (older csv), try to compute it from 'text'
    if "clean_text" not in df.columns:
        print("No 'clean_text' column found — using 'text' column instead.")
        df["clean_text"] = df["text"].astype(str)

    # Drop rows where label is missing
    before = len(df)
    df = df.dropna(subset=["label_num"])
    # Replace NaN/empty clean_text with empty string and then drop if still empty
    df["clean_text"] = df["clean_text"].astype(str)
    df["clean_text"] = df["clean_text"].replace("nan", "").replace(r"^\s*$", "", regex=True)
    df = df[df["clean_text"].str.strip() != ""]

    after = len(df)
    print(f"Dropped {before - after} rows with missing/empty text or label.")

    X = df["clean_text"]
    y = df["label_num"].astype(int)

    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Creating TF-IDF + Naive Bayes pipeline...")
    model = create_pipeline()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

    print(f"\nSaving model to {MODEL_PATH} ...")
    dump(model, MODEL_PATH)

    print("Done!")