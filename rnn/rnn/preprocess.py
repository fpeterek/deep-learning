import pandas as pd
import unicodedata
import re

from textblob import TextBlob
import nltk


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word) \
            .encode('ascii', 'ignore') \
            .decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_numbers(words):
    """Remove all interger occurrences in list of tokenized
    words with textual representation"""
    new_words = []
    for word in words:
        new_word = re.sub("\\d+", "", word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in nltk.corpus.stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = nltk.LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_numbers(words)
    words = remove_stopwords(words)
    return words


def form_sentence(text):
    text_blob = TextBlob(text)
    return text_blob.words


def fix_nt(words):
    st_res = []
    for i in range(0, len(words) - 1):
        if words[i+1] == "n't" or words[i+1] == "nt":
            st_res.append(words[i]+("n't"))
        else:
            if words[i] != "n't" and words[i] != "nt":
                st_res.append(words[i])
    return st_res


def join_to_string(text):
    return ' '.join(text)


def preprocess(ds) -> pd.DataFrame:
    ds['Rating'] = ds['Rating'].apply(lambda x: x - 1)
    ds['Review'] = ds['Review'].apply(form_sentence)
    ds['Review'] = ds['Review'].apply(normalize)
    ds['Review'] = ds['Review'].apply(fix_nt)
    ds['ReviewString'] = ds['Review'].apply(join_to_string)
    return ds
