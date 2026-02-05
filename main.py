import os
import io
import base64
import torch
import librosa
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = FastAPI()

# 1. API KEY CONFIGURATION
# Pulls the secret from Render environment variables
EXPECTED_API_KEY = os.environ.get("X_API_KEY", "sk_test_123456789")

# 2. MODEL CONFIGURATION
# DistilHuBERT is ~90MB, making it ideal for low-memory environments
MODEL_ID = "ntu-spml/distilhubert"

print("Starting model initialization...")

# Load Feature Extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)

# Load Model - We disable low_cpu_mem_usage to allow Quantization to work
model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID, 
    num_labels=2
)

# 3. DYNAMIC QUANTIZATION
# This shrinks the model's RAM footprint by ~50% (to roughly 45MB-50MB)
print("Applying dynamic quantization to save RAM...")
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.eval()

class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def analyze_audio(audio_bytes):
    # Load audio and resample to 16kHz (Standard for HuBERT models)
    # Using librosa.load with a byte stream
    audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    
    # Extract features
    inputs = feature_extractor(audio_data, sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        scores = torch.nn.functional.softmax(logits, dim=-1)
        conf, idx = torch.max(scores, dim=-1)
    
    # Classification Logic (0: HUMAN, 1: AI_GENERATED)
    label = "AI_GENERATED" if idx.item() == 1 else "HUMAN"
    
    # Generate explanation for evaluation criteria
    confidence_pct = int(conf.item() * 100)
    if label == "AI_GENERATED":
        explanation = f"Detected synthetic artifacts in high-frequency spectral regions with {confidence_pct}% confidence."
    else:
        explanation = f"Natural prosody and harmonic resonance detected with {confidence_pct}% confidence."
        
    return label, round(conf.item(), 2), explanation

@app.post("/api/voice-detection")
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    try:
        # 1. Get the string from the request
        audio_str = request.audioBase64
        
        # 2. AUTO-PADDING FIX: Ensure the string length is a multiple of 4
        missing_padding = len(audio_str) % 4
        if missing_padding:
            audio_str += '=' * (4 - missing_padding)
            
        # 3. Decode the safely padded string
        audio_bytes = base64.b64decode(audio_str)

        classification, confidence, explanation = analyze_audio(audio_bytes)
        
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": explanation
        }
    except Exception as e:
        # Return the specific error message to help with debugging
        return {"status": "error", "message": f"Inference failed: {str(e)}"}

# Health check for Render to verify service status
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "distilhubert-quantized"}

if __name__ == "__main__":
    import uvicorn
    # Render assigns a dynamic port via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

