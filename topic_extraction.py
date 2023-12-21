from bertopic import BERTopic
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

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
df = pd.DataFrame(pd.read_csv("comments_long.csv"))
l = list(df["Comments"])
processed_documents = preprocess_text(l)
print("Data loaded and preprocessed")

# We create an instance of the BERTopic model and fit it to the data. Note that we use the MiniLM embedding model as suggested in the library documentation
print("Fitting BERTopic model...")
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size = 30)
topics, _ = topic_model.fit_transform(processed_documents)
print("Model fitted")

#Plot the topics barchart
topic_model.visualize_barchart(title="Heat Pumps Comments Topics")

