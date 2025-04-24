from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import os
import traceback
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize app
app = FastAPI(title="Drone-Sensor Mission Recommender API")

# Load model and encoder
try:
    model = joblib.load("rf_mission_model.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Failed to load model:", e)

try:
    encoder = joblib.load("rf_label_encoder.joblib")
    print("✅ Label encoder loaded successfully")
except Exception as e:
    print("❌ Failed to load encoder:", e)

# Load features list
try:
    with open("rf_features.json") as f:
        FEATURES = json.load(f)
    print("✅ Features loaded:", FEATURES)
except Exception as e:
    print("❌ Failed to load features:", e)
    FEATURES = []

# Mount static HTML
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/form", response_class=HTMLResponse)
def serve_form():
    try:
        with open("templates/form.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading form</h1><p>{e}</p>", status_code=500)

# Input model
class MissionRequest(BaseModel):
    HazardType: int
    distance: float
    pop: float
    intensity: float
    duration_minutes: float
    economic_loss_million: float
    sensor_weight: float
    drone_speed: float
    drone_flight_time: float

@app.post("/recommend")
def recommend_mission(input_data: MissionRequest):
    try:
        input_dict = input_data.dict()
        print("🔹 Input received:", input_dict)

        # Confirm features
        df = pd.DataFrame([input_dict])
        print("🔹 Raw DataFrame:", df)

        # Check column ordering
        df = df[FEATURES]
        print("🔹 Reordered DataFrame:", df)

        # Check model type
        print("🔹 Model Type:", type(model))

        # Attempt prediction
        pred_class = model.predict(df)[0]
        print("🔹 Predicted Class:", pred_class)

        # Decode label
        pred_label = encoder.inverse_transform([pred_class])[0]
        print("🔹 Decoded Label:", pred_label)

        return {
            "recommended_drone_sensor_combo": pred_label,
            "input_summary": input_dict
        }

    except Exception as e:
        import traceback
        print("❌ EXCEPTION TRACE:")
        traceback.print_exc()
        return {"error": str(e)}

# Local testing only
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
