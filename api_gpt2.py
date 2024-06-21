from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(prompt: Prompt):
    inputs = tokenizer(prompt.prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"text": text}
