from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# Configuration du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser tous les domaines (à adapter en production)
    allow_credentials=True,
    allow_methods=["POST"],  # Autoriser uniquement la méthode POST
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

def load_model_and_tokenizer(model_name):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou du tokenizer: {e}")
        raise

# Chargement du modèle et du tokenizer une seule fois au démarrage du serveur
MODEL_NAME = "KingNish/Qwen2.5-0.5b-Test-ft"
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

def generate_response(model, tokenizer, prompt):
    try:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Erreur lors de la génération de la réponse: {e}")
        raise

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    try:
        response = generate_response(model, tokenizer, prompt_request.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

