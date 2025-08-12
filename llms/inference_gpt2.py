# load a small causal model with transformers for inference
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
input_ids = tok('Hello my name is', return_tensors='pt').input_ids
out = model.generate(input_ids, max_length=20)
print(tok.decode(out[0]))
