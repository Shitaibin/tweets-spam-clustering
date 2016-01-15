# Author: James Shi
# License: BSD 3 clause

# TODO: copy nltk_data from ubuntu and put into Anaconda\lib

from __future__ import print_function

import logging
import string

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('tweetspamtools')


#################################################
# Tfidf transform


def text_stem(text, stemmer):
    return [stemmer.stem(term) for term in text]


def tokenize(text):
    tokenized = word_tokenize(text)
    # removing punctation
    tokenized = [word for word in tokenized if word not in string.punctuation]
    stemmer = PorterStemmer()
    return text_stem(tokenized, stemmer)


def stem_tfidf_transform(documents):
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    transformed = vectorizer.fit_transform(documents)
    try:
        doc_vect = transformed.toarray()
        return doc_vect
    except Exception as e:
        logger.exception(e)
