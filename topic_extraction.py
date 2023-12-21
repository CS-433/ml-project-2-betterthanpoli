from bertopic import BERTopic
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# We start loading the spacy model we used for lemmaization and stopword removal
nlp = spacy.load("en_core_web_sm")

def preprocess_text(sentences):
    """
    Preprocesses the text by lemmatizing and removing stopwords

    Parameters:
    - sentences (list): List of sentences to preprocess

    Returns:
    - result (list): List of preprocessed sentences
    """
    result = []
    for sentence in sentences:
        doc = nlp(sentence.lower())
        lemmatized_tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS]
        result.append(' '.join(lemmatized_tokens))
    return result


# We load the data stored in a csv file and preprocess it
print("Loading and preprocessing data...")
df = pd.DataFrame(pd.read_csv("data/comments_long.csv"))
l = list(df["Comments"])
processed_documents = preprocess_text(l)
print("Data loaded and preprocessed")

# We create an instance of the BERTopic model and fit it to the data. Note that we use the MiniLM embedding model as suggested in the library documentation
print("Fitting BERTopic model...")
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size = 30)
topics, _ = topic_model.fit_transform(processed_documents)
print("Model fitted")


# we now define an helper function that can be used to create a wordcloud representation for a given topic
def create_wordcloud(topic_model, topic, ax, excluded_words = None, max_words = 8):
    if excluded_words is None:
        excluded_words = ["heat", "pump"]

    topic_words = topic_model.get_topic(topic)[0:10]

    # Filter out excluded words
    topic_words = [(word, value) for word, value in topic_words if word not in excluded_words]

    # Adjust the number of words based on the exclusion
    remaining_words = max_words - len(excluded_words)
    text = dict(topic_words[:remaining_words])
    wc = WordCloud(background_color="white", max_words=1000)
    wc.generate_from_frequencies(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
# From here, you can use create_wordcloud(topic_model,...) to produce any cloud you want.

