import pandas as pd
import matplotlib.pyplot as plt

def load_dataset():
    path = "data/raw/SMSSpamCollection"
    
    # Read tab-separated file
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    
    return df

def explore_data(df):
    print("First 5 rows:")
    print(df.head())
    print("\nDataset shape:", df.shape)
    
    print("\nClass distribution:")
    print(df['label'].value_counts())
    
    # Plot class distribution
    df['label'].value_counts().plot(kind='bar')
    plt.title("Spam vs Ham Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()

def check_missing(df):
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nDuplicate rows:", df.duplicated().sum())

if __name__ == "__main__":
    df = load_dataset()
    explore_data(df)
    check_missing(df)
