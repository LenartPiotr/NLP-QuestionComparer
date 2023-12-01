import pandas as pd
import numpy as np
import joblib
from os.path import join
import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def prepareCorpus(tolower = True, stemming = False, lemmatizing = True, dataFolder = '../data'):
    nltk.download('wordnet')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

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

    def custom_tokenize(text):
        tokens = casual_tokenize(text)
        if stemming:
            tokens = [stemmer.stem(token) for token in tokens]
        if lemmatizing:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        tokens = [token.lower() for token in tokens]
        return tokens

    print('Tokenize')
    items = list(corpus.values())
    model = TfidfVectorizer(tokenizer=custom_tokenize)
    data = model.fit_transform(np.array(items))

    print('Saving Tfidf model')
    joblib.dump(data, join(dataFolder, 'TfidfModel.model'))
    print('Done')

# joblib.load()