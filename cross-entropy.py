import random

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def cross_entropy(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    loss = outputs.loss.item()  # nats per token
    return loss / math.log(2)  # convert to bits per token


text_original = "The quick brown fox jumps over the lazy dog"


print("Token cross entropy:", cross_entropy(text_original))
print("-" * 60)

words = text_original.split()
random.shuffle(words)
shuffled_text = ' '.join(words)

print("Token cross entropy:", cross_entropy(shuffled_text))