import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def split(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def simple_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
