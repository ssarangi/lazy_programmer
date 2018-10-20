from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data = pd.read_csv('data/spambase/spambase.data').as_matrix()
np.random.shuffle(data)

X = data[:, 48]
Y = data[:, -1]

XTrain = X[:-100]
YTrain = Y[:-100]

XTest = X[-100:]
YTest = Y[-100:]

model = MultinomialNB()
model.fit(XTrain, YTrain)
print('Classification rate for NB: ', model.score(XTest, YTest))
