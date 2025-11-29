from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pandas as pd, numpy as np, joblib

app = FastAPI()

# Load the "Brain"
data = joblib.load("bike_model.pkl")
model = data['model']
scaler = data['scaler']
encoders = data['encoders']

@app.get("/", response_class=HTMLResponse)
def show_form():
    return """
    <html>
        <head>
            <title>Bike Renter Predictor</title>
            <style>
                body { font-family: sans-serif; max-width: 500px; margin: 40px auto; padding: 20px; }
                label { display: block; margin-top: 10px; font-weight: bold; }
                input, select { width: 100%; padding: 8px; margin-top: 5px; }
                button { margin-top: 20px; width: 100%; padding: 10px; background: #28a745; color: white; border: none; cursor: pointer; }
                .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; text-align: center; font-size: 1.2em; }
            </style>
        </head>
        <body>
            <h2>🚴 Bike Renter Prediction (KNN k=5)</h2>
            <form action="/predict" method="post">
                <label>Season:</label>
                <select name="Season">
                    <option value="Winter">Winter</option><option value="Spring">Spring</option>
                    <option value="Summer">Summer</option><option value="Fall">Fall</option>
                </select>

                <label>Month (1-12):</label>
                <input type="number" name="Month" required min="1" max="12">

                <label>Weekday:</label>
                <select name="Weekday">
                    <option value="Monday">Monday</option><option value="Tuesday">Tuesday</option>
                    <option value="Wednesday">Wednesday</option><option value="Thursday">Thursday</option>
                    <option value="Friday">Friday</option><option value="Saturday">Saturday</option>
                    <option value="Sunday">Sunday</option>
                </select>

                <label>Working Day (0 = No, 1 = Yes):</label>
                <select name="Working_Day"><option value="1">Yes (1)</option><option value="0">No (0)</option></select>

                <label>Temperature (C):</label>
                <input type="number" step="0.1" name="Temp" required>

                <label>Humidity (%):</label>
                <input type="number" step="0.1" name="Humidity" required>

                <label>Wind (km/h):</label>
                <input type="number" step="0.1" name="Wind" required>

                <button type="submit">Predict Renters</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(
    Season: str = Form(...), Month: int = Form(...), Weekday: str = Form(...),
    Working_Day: int = Form(...), Temp: float = Form(...), Humidity: float = Form(...), Wind: float = Form(...)
):
    try:
        # Prepare Data Frame
        input_dict = {
            'Season': [Season], 'Month': [Month], 'Weekday': [Weekday],
            'Working Day': [Working_Day], 'Temp C': [Temp],
            'Humidity %': [Humidity], 'Wind km/h': [Wind]
        }
        df_in = pd.DataFrame(input_dict)

        # Apply Saved Encoders
        for col, le in encoders.items():
            df_in[col] = le.transform(df_in[col])

        # Apply Saved Scaler
        X_final = scaler.transform(df_in)

        # Predict
        pred = model.predict(X_final)[0]
        
        # Binning Logic for Result Display
        # Based on your data: Low < ~600, Mid ~600-1400, High > 1400 (Approx)
        category = "Low" if pred < 600 else "High" if pred > 1400 else "Medium"

        return f"""
        <html><body>
            <div style="font-family: sans-serif; max-width: 500px; margin: 40px auto; text-align: center;">
                <h1>Prediction Result</h1>
                <div style="font-size: 2em; color: #007bff; font-weight: bold;">{int(pred)} Renters</div>
                <p>Demand Level: <strong>{category}</strong></p>
                <a href="/">Make another prediction</a>
            </div>
        </body></html>
        """
    except Exception as e:
        return f"Error: {str(e)}"