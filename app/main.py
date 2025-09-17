from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from .model.model_def import SimpleDNA_CNN
from sklearn.preprocessing import LabelEncoder
from .process import one_hot_encode
import torch

class InputSequence(BaseModel):
    sequence : str
    
class OutputPrediction(BaseModel):
    prediction : str

# Create an instance of the FastAPI class
app = FastAPI()

# Build a robust path to the model file
BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "DeepTide_v0.pth"
MODEL_PATH = BASE_DIR / "model" / MODEL_NAME


model = SimpleDNA_CNN()

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()
classes = ['Cnidaria', 'Arthropoda', 'Porifera', 'Echinodermata']
label_encoder = LabelEncoder()
label_encoder.fit_transform(classes)
CLASSES = label_encoder.classes_
print(CLASSES)

# Define a path operation decorator for the root endpoint
@app.get("/")
def home():
    return {"Hello": "World"}

# Define another endpoint
@app.post("/predict", response_model=OutputPrediction, status_code=200)
def predict(request: InputSequence):
    EXPECTED_LENGTH = 300
    if len(request.sequence) != EXPECTED_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sequence length. Expected {EXPECTED_LENGTH} characters, but got {len(request.sequence)}."
        )
    encoded_seq = one_hot_encode(request.sequence)
    input_tensor = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0).to("cpu")

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    # Decode the predicted class
    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())
    print("Predicted class:", predicted_label[0])

    return {"prediction": predicted_label[0]}

