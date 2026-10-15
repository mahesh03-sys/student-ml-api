from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Student Performance API")

# Load the trained model
model = joblib.load('model.pkl')

# Input data format
class StudentData(BaseModel):
    study_hours: float
    attendance: float
    prev_marks: float

@app.get("/health")
def health_check():
    return {"status": "API is running perfectly"}

@app.post("/predict")
def predict_performance(data: StudentData):
    input_features = [[data.study_hours, data.attendance, data.prev_marks]]
    prediction = model.predict(input_features)
    return {"predicted_score": round(prediction[0], 2)}