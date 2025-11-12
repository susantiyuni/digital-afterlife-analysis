import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def run_bertopic(input_file: str = "ENDEVR_clean.csv", output_file: str = "ENDEVR_btopic.csv"):
    df = pd.read_csv(input_file)
    stop_words = stopwords.words("english")

    vectorizer = CountVectorizer(stop_words=stop_words)
    topic_model = BERTopic(
        vectorizer_model=vectorizer,
        language="english",
        verbose=True,
        calculate_probabilities=False
    )

    topics, _ = topic_model.fit_transform(df["clean"])
    df["topic"] = topics
    df.to_csv(output_file, index=False)

    topic_info = topic_model.get_topic_info()
    print("\nTop 10 Topics:\n")
    print(topic_info.head(10))

    print(f"✅ BERTopic results saved to '{output_file}'")

if __name__ == "__main__":
    run_bertopic()
