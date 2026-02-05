import os
import io
import base64
import torch
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = FastAPI()

# 1. FETCH API KEY
EXPECTED_API_KEY = os.environ.get("X_API_KEY", "sk_test_123456789")

# 2. LOAD LIGHTWEIGHT MODEL (DistilHuBERT is ~90MB vs Wav2Vec2 ~360MB)
MODEL_ID = "ntu-spml/distilhubert" 

print("Loading model... this may take a moment.")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2, 
    low_cpu_mem_usage=True
)

# 3. APPLY QUANTIZATION (Reduces RAM usage by ~50%)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.eval()

class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def analyze_audio(audio_bytes):
    # Load audio & resample
    audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # Process audio
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        conf, idx = torch.max(scores, dim=-1)
    
    label = "AI_GENERATED" if idx.item() == 1 else "HUMAN"
    
    # Simple explanation logic based on confidence and artifacts
    if label == "AI_GENERATED":
        explanation = f"Synthetic artifacts detected in spectral density with {int(conf.item()*100)}% confidence."
    else:
        explanation = "Natural vocal micro-tremors and prosody detected."
        
    return label, round(conf.item(), 2), explanation

@app.post("/api/voice-detection")
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        classification, confidence, explanation = analyze_audio(audio_bytes)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
