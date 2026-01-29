import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model
model_path = "resnet_model_finetuned2.h5"  # Ensure this file exists in the correct path
model = load_model(model_path)
print("✅ Model Loaded Successfully!")

# Define test dataset path
test_dir = "D:/animal/dataset/test"  # Ensure this folder contains subfolders for each class

# Data Preprocessing
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Get true labels and class indices
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Predict on test data
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot Confusion Matrix with Fixes
plt.figure(figsize=(15, 12))  # Increase figure size
sns.heatmap(conf_matrix, annot=np.where(conf_matrix > 0, conf_matrix, ""), fmt="s", cmap="Blues")

plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix for Animal Recognition", fontsize=14)
plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=90, fontsize=10)
plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0, fontsize=10)

# Save the confusion matrix as a high-resolution image
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_labels))
