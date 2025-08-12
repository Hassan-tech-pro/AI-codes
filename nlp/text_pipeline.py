# simple text classification with transformers pipeline
from transformers import pipeline
clf = pipeline('sentiment-analysis')
print(clf('I love machine learning'))
