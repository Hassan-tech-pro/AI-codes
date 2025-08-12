# note this is a template for finetune with datasets and Trainer
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

ds = load_dataset('imdb', split='train[:1%]')
tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
def prep(ex):
    return tok(ex['text'], truncation=True, padding='max_length')
ds = ds.map(prep, batched=True)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

args = TrainingArguments(output_dir='./out', per_device_train_batch_size=8, num_train_epochs=1)
trainer = Trainer(model=model, args=args, train_dataset=ds)
# trainer.train()  # run this in a machine with GPU
print('template ready')
