from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from pathlib import Path
from .model.model_def import SimpleDNA_CNN, FeatureExtractor
from sklearn.preprocessing import LabelEncoder
from .process import one_hot_encode, compute_prediction, extract_fasta
import torch
import io
from typing import List, Dict, Any
from pathlib import Path

class InputSequence(BaseModel):
    sequence : str
    
class OutputPrediction(BaseModel):
    prediction : str
    confidence : float

# Create an instance of the FastAPI class
app = FastAPI()

# Build a robust path to the model file
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "DeepTide_v0.pth"
MODEL_PATH = BASE_DIR / "model" / MODEL_NAME
DEVICE = torch.device('cpu');

model = SimpleDNA_CNN()

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

classes = ['Cnidaria', 'Arthropoda', 'Porifera', 'Echinodermata']
label_encoder = LabelEncoder()
label_encoder.fit_transform(classes)

feature_extractor = FeatureExtractor(model).to(DEVICE)
feature_extractor.eval();

EXPECTED_LENGTH = 300

@app.get("/")
def home():
    return {"message": "Welcome to the Sequence Prediction API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=OutputPrediction, status_code=200)
async def predict(request: InputSequence):
    """Predicts the class for a single DNA sequence."""
    sequence = request.sequence
    if len(sequence) != EXPECTED_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sequence length. Expected {EXPECTED_LENGTH}, but got {len(sequence)}."
        )

    try:
        encoded_seq = one_hot_encode(sequence)
        input_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        prediction, confidence = compute_prediction(model, input_tensor, label_encoder)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    return {"prediction": prediction, "confidence": confidence}


@app.post("/bulk_predict", status_code=200)
async def bulk_predict(file: UploadFile = File(...)):
    """Performs batch prediction from a FASTA file."""
    if not file.filename or not file.filename.endswith(('.fasta', '.fa', '.fna')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a FASTA file.")
    
    try:
        text_stream = io.TextIOWrapper(file.file, encoding="utf-8")
        df = extract_fasta(text_stream)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse FASTA file: {str(e)}")

    if df.empty:
        return []

    output: List[Dict[str, Any]] = []

    for rec in df.to_dict(orient="records"):
        sequence = rec.get("Sequence", "")
        seq_id = rec.get("ID", "Unknown")
        result: Dict[str, Any] = {"id": seq_id}

        if len(sequence) != EXPECTED_LENGTH:
            result["error"] = f"Invalid sequence length. Expected {EXPECTED_LENGTH}, but got {len(sequence)}."
        else:
            try:
                encoded_seq = one_hot_encode(sequence)
                input_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                prediction, confidence = compute_prediction(model, input_tensor, label_encoder)
                result["prediction"] = prediction
                result["confidence"] = confidence
            except Exception as e:
                result["error"] = f"Model prediction failed: {str(e)}"
        
        output.append(result)

    return output

