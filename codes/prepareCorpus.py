import pandas as pd
import numpy as np
import joblib
from os.path import join
import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemming = False
lemmatizing = True

def custom_tokenize(text):
        tokens = casual_tokenize(text)
        if stemming:
            tokens = [stemmer.stem(token) for token in tokens]
        if lemmatizing:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token.lower() for token in tokens]
        return tokens

def prepareCorpus(tolower = True, _stemming = False, _lemmatizing = True, dataFolder = '../data', max_features=200000):

    stemming = _stemming
    lemmatizing = _lemmatizing

    print('Reading data')
    train = pd.read_csv(join(dataFolder, 'originals/train.csv'))

    print('Building corpus')
    corpus = {}
    for i in range(len(train)):
        for collId in range(1,3):
            qid = train['qid' + str(collId)][i]
            quest = train['question' + str(collId)][i]
            if qid not in corpus:
                corpus[qid] = quest if tolower else quest

    print('Tokenize')
    items = list(corpus.values())
    model = TfidfVectorizer(tokenizer=custom_tokenize, max_features=max_features)
    data = model.fit_transform(np.array(items))
    joblib.dump(model, join(dataFolder, 'TfidfVectorizer.pkl'))

    print('Merging')
    train = pd.merge(train, pd.DataFrame({'qid1': range(1, data.shape[0] + 1), 'tfidf1': data}), on='qid1', how='left')
    train = pd.merge(train, pd.DataFrame({'qid2': range(1, data.shape[0] + 1), 'tfidf2': data}), on='qid2', how='left')
    train['tfidf'] = train.apply(lambda row: hstack([row['tfidf1'], row['tfidf2']]), axis=1)

    print('Saving Tfidf model')
    joblib.dump(train, join(dataFolder, 'TfidfDataframe.pkl'))
    print('Done')

# joblib.load()