import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model("models/final_best_model.h5")
scaler = joblib.load("models/scaler.pkl")

input_file = "dataset/test_data.csv" 
df = pd.read_csv(input_file)

expected_cols = ["Acc X", "Acc Y", "Acc Z", "gyro_x", "gyro_y", "gyro_z"]

X = df[expected_cols]
X_scaled = scaler.transform(X)

y_pred_probs = model.predict(X_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

label_map = {0: "normal", 1: "agresiv", 2: "defensiv"}
df["predicted_label"] = [label_map[i] for i in y_pred]

print("\nPredicted Class Counts:")
print(df["predicted_label"].value_counts())
