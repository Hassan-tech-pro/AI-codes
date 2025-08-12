# tokenizer example
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print(tok('hello world', return_tensors='pt'))
