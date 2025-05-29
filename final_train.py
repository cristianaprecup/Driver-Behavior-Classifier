import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib  # for saving the scaler

print("Loading dataset...")
df = pd.read_csv("dataset/3_FinalDatasetCsv.csv")

print(">>> Extracting features and labels...")
X = df[["Acc X", "Acc Y", "Acc Z", "gyro_x", "gyro_y", "gyro_z"]]
y = to_categorical(df["label"], num_classes=3)

print(">>> Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(">>> Saving scaler to 'models/scaler.pkl'...")
joblib.dump(scaler, "models/scaler.pkl")

print(">>> Building model with architecture: [109, 57]...")
model = Sequential()
model.add(Dense(109, input_shape=(6,), activation="relu"))
model.add(Dense(57, activation="relu"))
model.add(Dense(3, activation="softmax"))

print(">>> Compiling model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

print(">>> Training model (30 epochs)...")
model.fit(X_scaled, y, epochs=30, batch_size=32, verbose=1)

print(">>> Saving trained model to 'models/final_best_model.h5'...")
model.save("models/final_best_model.h5")

print(">>> DONE: Model and scaler saved successfully.")

from sklearn.metrics import confusion_matrix, classification_report

# Predict on all training data
y_pred_prob = model.predict(X_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y, axis=1)

# Print evaluation metrics
print("\n>>> Model Evaluation:")
print("Accuracy on full dataset: {:.2f}%".format((y_pred == y_true).mean() * 100))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "agresiv", "defensiv"]))


