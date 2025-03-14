import joblib
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

# Define dataset path
path = 'C:\\Users\\HP India\\Documents\\BrainTumorWebsite\\Training'
classes = {'no_tumor': 0, 'pituitary_tumor': 1}

X = []
Y = []

# Load and preprocess images
for cls in classes:
    cls_path = os.path.join(path, cls)
    for img_name in os.listdir(cls_path):
        img = cv2.imread(os.path.join(cls_path, img_name), 0)  # Read in grayscale
        img = cv2.resize(img, (200, 200))  # Resize
        X.append(img)
        Y.append(classes[cls])

# Convert to NumPy arrays
X = np.array(X)
Y = np.array(Y)

# Reshape and normalize
X_updated = X.reshape(len(X), -1) / 255.0

# Split into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(xtrain, ytrain)

# Save the model
joblib.dump(svm_model, 'svm_model.pkl')
print("SVM model saved as 'svm_model.pkl'.")
