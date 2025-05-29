import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

def generate_all_plots(X_scaled, y_true, y_pred, df, history=None):
    labels = ["normal", "agresiv", "defensiv"]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Driving Style")
    plt.ylabel("Actual Driving Style")
    plt.title("Confusion Matrix\n(Comparison between model predictions and real behavior)")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=labels)
    print(report)

    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    precision = [report_dict[cls]["precision"] for cls in labels]
    recall = [report_dict[cls]["recall"] for cls in labels]
    f1 = [report_dict[cls]["f1-score"] for cls in labels]
    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar([p - width for p in x], precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar([p + width for p in x], f1, width, label="F1-score")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Classification Metrics per Class\n(Precision, Recall, F1-Score)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if history:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.title("Training Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(6, 4))
    df["label"].value_counts().sort_index().plot(kind="bar", tick_label=labels)
    plt.title("Driving Style Distribution in Dataset")
    plt.xlabel("Driving Style")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()

    features = ["Acc X", "Acc Y", "Acc Z", "gyro_x", "gyro_y", "gyro_z"]
    feature_explanations = {
        "Acc X": "Acceleration X (longitudinal)",
        "Acc Y": "Acceleration Y (lateral)",
        "Acc Z": "Acceleration Z (vertical)",
        "gyro_x": "Gyroscope X (roll rate)",
        "gyro_y": "Gyroscope Y (pitch rate)",
        "gyro_z": "Gyroscope Z (yaw rate)"
    }

    for feat in features:
        plt.figure()
        for cls, name in zip([0, 1, 2], labels):
            subset = df[df["label"] == cls]
            plt.hist(subset[feat], bins=50, alpha=0.5, label=name)
        plt.title(f"{feature_explanations[feat]} by Driving Style")
        plt.xlabel(feature_explanations[feat] + " [normalized]")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    df_vis = df.copy()
    df_vis["label"] = df_vis["label"].map({0: "normal", 1: "agresiv", 2: "defensiv"})
    sns.pairplot(df_vis[["Acc X", "Acc Y", "Acc Z", "label"]], hue="label")
    plt.suptitle("Pairwise Feature Relationships (Acceleration only)", y=1.02)
    plt.show()