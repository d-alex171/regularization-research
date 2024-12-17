import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


def preprocess_text_nltk(text, lowercase=True, remove_stopwords=True, remove_punctuation=True):
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')


    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Process tokens
    processed_tokens = []
    for token in tokens:
        if remove_stopwords and token in stop_words:
            continue
        if remove_punctuation and token in string.punctuation:
            continue
        # Lemmatize the token
        processed_tokens.append(lemmatizer.lemmatize(token))

    preprocessed_text = " ".join(processed_tokens)
    return preprocessed_text