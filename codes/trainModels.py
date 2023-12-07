from os.path import join
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity

def trainModelTfIdf(model, modelName, dataFolder = '../data', training = True, predicting = True):
    print('Reading tfidf model')
    df = joblib.load(join(dataFolder, 'TfidfDataframe.pkl'))

    if training:
        print('Preparing train and test data')
        X = df['tfidf'].tolist()
        y = df['is_duplicate'].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f'Training {modelName} model')

        model.fit(vstack(X_train), y_train)

        print('Saving')
        joblib.dump(model, join(dataFolder, f'{modelName}.pkl'))
    else:
        model = joblib.load(join(dataFolder, f'{modelName}.pkl'))

    if predicting:
        if not training:
            X_test = df['tfidf'].tolist()
            y_test = df['is_duplicate'].tolist()
        print('Predicting tests')
        y_pred = model.predict(vstack(X_test))

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')

        report = classification_report(y_test, y_pred)
        print('Classification Report:\n', report)

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot()


def cosineSimilarity(dataFolder = '../data'):
    print('Reading tfidf model')
    df = joblib.load(join(dataFolder, 'TfidfDataframe.pkl'))

    X1_test = vstack(df['tfidf1'].tolist())
    X2_test = vstack(df['tfidf2'].tolist())
    y_test = df['is_duplicate'].tolist()

    print('Predicting tests')
    y_pred = [cosine_similarity(X1_test[i], X2_test[i])[0, 0] for i in range(X1_test.shape[0])]
    y_pred = np.array(y_pred)
    y_pred = y_pred > 0.5

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    report = classification_report(y_test, y_pred)
    print('Classification Report:\n', report)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()