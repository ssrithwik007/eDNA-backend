from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from .model.model_def import SimpleDNA_CNN, FeatureExtractor
from sklearn.preprocessing import LabelEncoder
from .process import (
    one_hot_encode, compute_prediction, extract_fasta, process_sequences, create_umap
)
import torch
import io
import pandas as pd
import joblib 
from typing import List, Dict, Any
import numpy as np 

class InputSequence(BaseModel):
    sequence: str
    
class OutputPrediction(BaseModel):
    prediction: str
    confidence: float

# Create an instance of the FastAPI class
app = FastAPI()

# --- Paths and Constants ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "DeepTide_v0.pth"
MODEL_PATH = BASE_DIR / "model" / MODEL_NAME
REF_DATA_PATH = BASE_DIR / "reference_map_data.csv"
UMAP_REDUCER_PATH = BASE_DIR / "umap_reducer_model.joblib"
DEVICE = torch.device('cpu')
SEQ_LENGTH = 300

# --- Load Models and Data on Startup ---
# Load predictive model
model = SimpleDNA_CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Setup label encoder
classes = ['Cnidaria', 'Arthropoda', 'Porifera', 'Echinodermata']
label_encoder = LabelEncoder()
label_encoder.fit_transform(classes)

# Load feature extractor
feature_extractor = FeatureExtractor(model).to(DEVICE)
feature_extractor.eval()

# Load reference UMAP data for the background plot
reference_df = pd.read_csv(REF_DATA_PATH)

# Load the pre-fitted UMAP reducer model
umap_reducer = joblib.load(UMAP_REDUCER_PATH)

@app.get("/")
def home():
    return {"message": "Welcome to the Sequence Prediction API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=OutputPrediction, status_code=200)
async def predict(request: InputSequence):
    """Predicts the class for a single DNA sequence."""
    sequence = request.sequence[:SEQ_LENGTH].ljust(SEQ_LENGTH, 'N')

    try:
        encoded_seq = one_hot_encode(sequence)
        input_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prediction, confidence = compute_prediction(model, input_tensor, label_encoder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    return {"prediction": prediction, "confidence": confidence}


@app.post("/bulk_predict", status_code=200)
async def bulk_predict(file: UploadFile = File(...)):
    """Performs batch prediction from a FASTA file and overlays it on a reference UMAP."""
    # ... (file validation and parsing logic remains the same) ...
    if not file.filename or not file.filename.endswith(('.fasta', '.fa', '.fna')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a FASTA file.")
    try:
        text_stream = io.TextIOWrapper(file.file, encoding="utf-8")
        df = extract_fasta(text_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse FASTA file: {str(e)}")
    if df.empty:
        return JSONResponse(content={"summary": {}, "results": [], "plot": ""})

    # ... (prediction loop and summary calculation remain the same) ...
    predictions: List[Dict[str, Any]] = []
    prediction_labels = []
    class_counts = {cls: 0 for cls in classes}
    confidence_sum = 0
    valid_predictions = 0
    for rec in df.to_dict(orient="records"):
        sequence = rec.get("Sequence", "")[:SEQ_LENGTH].ljust(SEQ_LENGTH, 'N')
        seq_id = rec.get("ID", "Unknown")
        result: Dict[str, Any] = {"id": seq_id}
        try:
            encoded_seq = one_hot_encode(sequence)
            input_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prediction, confidence = compute_prediction(model, input_tensor, label_encoder)
            result["prediction"] = prediction
            result["confidence"] = confidence
            prediction_labels.append(prediction)
            class_counts[prediction] += 1
            confidence_sum += confidence
            valid_predictions += 1
        except Exception as e:
            prediction_labels.append("N/A")
            result["error"] = f"Model prediction failed: {str(e)}"
        predictions.append(result)
    
    avg_confidence = confidence_sum / valid_predictions if valid_predictions > 0 else 0
    summary = {
        "class_counts": class_counts, "average_confidence": round(avg_confidence, 4),
        "total_sequences": len(predictions), "valid_predictions": valid_predictions
    }

    # --- Generate Plot ---
    cluster_df, embeddings = process_sequences(df["Sequence"], feature_extractor, device="cpu")

    # Call create_umap with the new required arguments
    plot_base64 = create_umap(
        embeddings=embeddings,
        cluster_labels=cluster_df['cluster_label'],
        predictions=prediction_labels
    )

    # --- Final Response ---
    final_response = {
        "summary": summary,
        "results": predictions,
        "cluster_labels": cluster_df.to_dict(orient="records"),
        "plot": plot_base64
    }

    return JSONResponse(content=final_response)