from os.path import join
import numpy as np
import pandas as pd
import joblib

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def trainMultinomialNB(dataFolder = '../data'):
    print('Reading tfidf model')
    tfidfData = joblib.load(join(dataFolder, 'TfidfModel.model'))
    df = pd.read_csv(join(dataFolder, 'originals/train.csv'))
    
    print('Preparing train and test data')
    X = df
    y = df['is_duplicate'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training MultinomialNB model')
    model = MultinomialNB()

    lastPercent = 0
    for i in range(len(X_train)):
        x_data = np.array([np.concatenate([tfidfData[X_train['qid1'].iloc[i] - 1].toarray()[0], tfidfData[X_train['qid2'].iloc[i] - 1].toarray()[0]])])
        # print(x_data.shape)
        model.partial_fit(x_data, [y_train[i]], classes=np.unique(y_train))
        percent = round(i / len(X_train) * 100)
        if percent > lastPercent:
            print(str(percent) + '%')
            lastPercent = percent

    print('Saving')
    joblib.dump(model, join(dataFolder, 'MultinomialNB.model'))

    print('Predicting tests')
    y_pred = []
    for i in range(len(X_test)):
        y_pred.extend([model.predict(np.concatenate([tfidfData[X_test['qid1'].iloc[i] - 1].toarray()[0], tfidfData[X_test['qid2'].iloc[i] - 1].toarray()[0]]))])

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    report = classification_report(y_test, y_pred)
    print('Classification Report:\n', report)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
