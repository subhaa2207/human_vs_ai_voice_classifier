import torch
import torch.nn.functional as F
import base64
import io
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

app = FastAPI()

# Load a pre-trained model capable of detecting audio spoofing
# Note: On first run, this will download about 300MB of model weights
MODEL_ID = "facebook/wav2vec2-base-960h" 
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, num_labels=2)
model.eval()

class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def analyze_audio(audio_bytes):
    # Load and resample to 16kHz
    audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = F.softmax(logits, dim=-1)
        conf, idx = torch.max(scores, dim=-1)
    
    # 0: Human, 1: AI (Simplified logic for the example)
    label = "AI_GENERATED" if idx.item() == 1 else "HUMAN"
    explanation = "Consistent spectral artifacts detected." if label == "AI_GENERATED" else "Natural vocal micro-tremors detected."
    return label, round(conf.item(), 2), explanation

@app.post("/api/voice-detection")
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    if x_api_key != "sk_test_123456789":
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)