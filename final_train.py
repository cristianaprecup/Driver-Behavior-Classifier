import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib  

df = pd.read_csv("dataset/3_FinalDatasetCsv.csv")

X = df[["Acc X", "Acc Y", "Acc Z", "gyro_x", "gyro_y", "gyro_z"]]
y = to_categorical(df["label"], num_classes=3)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

model = Sequential()
model.add(Dense(109, input_shape=(6,), activation="relu"))
model.add(Dense(57, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

print("training model 30 epochs...")
history = model.fit(X_scaled, y, epochs=30, batch_size=32, verbose=1)
model.save("models/final_model.h5")

from sklearn.metrics import confusion_matrix, classification_report

y_pred_prob = model.predict(X_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y, axis=1)

print("\n Model Evaluation:")
print("Accuracy on full dataset: {:.2f}%".format((y_pred == y_true).mean() * 100))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "agresiv", "defensiv"]))

import plots
plots.generate_all_plots(X_scaled, y_true, y_pred, df, history)

