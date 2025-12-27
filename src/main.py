import joblib
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Custom Sentiment API")

# 1. Load your trained model
# We load it once when the app starts.
mode_path = "./out/sentiment_model.pkl"
try:
    model = joblib.load(mode_path)
    print("Custom model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    probabilities: dict

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 2. Make prediction
    # The pipeline handles vectorization automatically
    prediction = model.predict([request.text])[0]
    
    # 3. Get probabilities (confidence)
    # model.predict_proba returns an array like [[0.1, 0.9]] (negative, positive)
    probs = model.predict_proba([request.text])[0]
    classes = model.classes_ # e.g., ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    
    # Create a clean dictionary of scores
    prob_dict = {classes[i]: round(probs[i], 4) for i in range(len(classes))}

    return {
        "label": prediction,
        "probabilities": prob_dict
    }