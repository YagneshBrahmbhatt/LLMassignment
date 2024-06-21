# LLMassignment
Mini Assignment: Local LLM Deployment and Interaction

# Llama3 and GPT APIs Setup Guide
This guide provides step-by-step instructions to set up and interact with Llama3, GPT-Neo, and GPT-2 models using FastAPI and curl commands.

## Step 1: Download and Run Llama3
# Prerequisites
Before downloading and running Llama3, ensure you have the following tools and dependencies installed on your system:

Git (for cloning repositories)
Python (version 3.6 or higher)
Docker (if running models in a containerized environment)

# Clone the Llama3 Repository
Open your terminal (Command Prompt, PowerShell, Terminal, etc.).
Clone the Llama3 repository from GitHub:

```
git clone https://github.com/Ollama/llama3.git
```

# Install Dependencies
Navigate into the cloned repository directory:
```
cd llama3
```
Install necessary dependencies, typically managed using Python's package manager, pip:
```
pip install -r requirements.txt
```

## Step 2: Setup FastAPI for GPT Models
# Install FastAPI and Uvicorn

Ensure FastAPI and Uvicorn are installed for serving GPT models via HTTP:
```
pip install fastapi uvicorn
```
# Create APIs for GPT-Neo and GPT-2
# GPT-Neo API Setup (api_gpt_neo.py)
Create a Python file named api_gpt_neo.py with the following content:
```
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.post("/")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100)
    return {"text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
```

# GPT-2 API Setup (api_gpt2.py)
Create another Python file named api_gpt2.py for GPT-2 with similar content, adjusting the model as necessary.

## Run FastAPI Servers
Open two terminal windows or tabs.
Start the FastAPI server for GPT-Neo:
```
uvicorn api_gpt_neo:app --host 0.0.0.0 --port 8000 --reload
```
Start the FastAPI server for GPT-2 in the second terminal:
```
uvicorn api_gpt2:app --host 0.0.0.0 --port 8001 --reload
```
## Using curl to Interact with the APIs
You can interact with the APIs using curl commands. Here's an example:
```
curl -X POST "http://localhost:8000" -H "Content-Type: application/json" -d "{\"prompt\": \"Once upon a time in a land far, far away,\"}"
```
Replace "http://localhost:8000" with the appropriate URL for the API endpoint you wish to interact with.

## Repository Structure
api_gpt_neo.py: Script to generate a short story using the GPT-Neo model.
api_gpt2.py: Script to generate a conversation using the GPT-2 model.
README.md: Documentation for setting up and running the scripts.

# Notes
Ensure you have a stable internet connection when running the scripts for the first time, as the models and tokenizers need to be downloaded.
The scripts are set to generate text based on simple prompts. You can modify the prompts and other parameters (like max_length and num_return_sequences) to suit your requirements.

## Conclusion
This repository provides a simple and effective way to interact with local language models for generating text. Feel free to explore and modify the scripts to create more complex and interesting outputs.

If you encounter any issues or have suggestions for improvements, please feel free to create an issue or submit a pull request.

Happy coding!

