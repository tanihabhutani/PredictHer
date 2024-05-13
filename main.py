from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

class DiabeticsPatient(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int

class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float  

class BreastCancer(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float

# Load the trained machine learning model
diabetics_model = joblib.load('diabetes_model.pkl')
heart_disease_model = joblib.load('heart_disease_model.pkl')
breast_cancer_model = joblib.load('breast_cancer.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/predict_diabetes")
async def predict_diabetes(patient: DiabeticsPatient):

    input_data = [[patient.pregnancies, patient.glucose, patient.blood_pressure,
                   patient.skin_thickness, patient.insulin, patient.bmi,
                   patient.diabetes_pedigree, patient.age]]
    
    prediction = diabetics_model.predict(input_data)
    
    return {"prediction": bool(prediction[0])}

@app.post("/predict_heart_disease")
def predict_heart_disease(data: HeartDiseaseInput):
    input_data = [[data.age,data.sex, data.cp,
                   data.trestbps, data.chol, data.fbs,
                   data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]]
    
    prediction = heart_disease_model.predict(input_data)
    
    return {"prediction": bool(prediction)}

@app.post("/predict_breast_cancer")
async def predict_diabetes(patient: BreastCancer):

    input_data = [[patient.mean_radius, patient.mean_texture, patient.mean_perimeter, patient.mean_area, patient.mean_smoothness]]
    
    prediction = breast_cancer_model.predict(input_data)
    
    return {"prediction": bool(prediction[0])}


