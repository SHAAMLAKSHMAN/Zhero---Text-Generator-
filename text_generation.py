# text_generation.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# 1. Verify imports work first
print("Libraries imported successfully!")

# 2. Sample data
data = {"text": [
    "To be or not to be, that is the question.",
    "All the world's a stage.",
    "Hello, world!"
]}

# 3. Rest of your code...

# Prepare dataset
dataset = Dataset.from_pandas(pd.DataFrame(data))
dataset = dataset.train_test_split(test_size=0.2)

# Load model/tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenization
def tokenize(batch):
    tokenized = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()
model.save_pretrained("./gpt2-finetuned")

from transformers import pipeline

# Load fine-tuned model
generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="gpt2")

# Generate text
print(generator("AI will change the world by", max_length=50))

import gradio as gr
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="gpt2")

def generate(prompt):
    return generator(prompt, max_length=100)[0]["generated_text"]

gr.Interface(fn=generate, inputs="text", outputs="text").launch()

