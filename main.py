import torch
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# prepare the dataset and split into train and test subsections
dataset = pd.read_csv('data/df_file.csv').to_numpy()
X = dataset[:, 0]
y = dataset[:, 1]
enc = CountVectorizer()
X = enc.fit_transform(X).toarray()
dataset = np.c_[X, y]
print(dataset)
train_test_split(dataset, shuffle=True, test_size=0.3)

