import torch
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

from preprocessor import preprocess_text_nltk
import obj_fun

# prepare the dataset and split into train and test subsections
dataset = pd.read_csv('data/df_file.csv').to_numpy()
for i in range(dataset.shape[0]):
    dataset[i, 0] = preprocess_text_nltk(dataset[i, 0])
X = dataset[:, 0]
y = dataset[:, 1]
print(X)
enc = CountVectorizer()
X = enc.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
print(y_train)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
n, d = X_train.shape


classifier = obj_fun.SoftmaxClassifierL2(num_classes=5, num_features=d)
classifier.train(X_train, y_train, epochs=1000)
y_pred = classifier.predict(X_test)
print(f"L2 Accuracy is {np.mean(y_pred == y_test)}")

classifier = obj_fun.SoftmaxClassifierL1(num_classes=5, num_features=d)
classifier.train(X_train, y_train, epochs=1000)
y_pred = classifier.predict(X_test)
print(f"L1 Accuracy is {np.mean(y_pred == y_test)}")

classifier = obj_fun.SoftmaxClassifierL0(num_classes=5, num_features=d)
classifier.train(X_train, y_train, epochs=1000)
y_pred = classifier.predict(X_test)
print(f"L0 Accuracy is {np.mean(y_pred == y_test)}")

classifier = obj_fun.SoftmaxClassifierElasticNet(num_classes=5, num_features=d)
classifier.train(X_train, y_train, epochs=1000)
y_pred = classifier.predict(X_test)
print(f"ElasticNet Accuracy is {np.mean(y_pred == y_test)}")