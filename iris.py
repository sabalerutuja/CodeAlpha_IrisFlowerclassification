# Iris Flower Classification using RandomForestClassifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# 1. Load the dataset
iris_df = pd.read_csv("Iris.csv")   # Make sure Iris.csv is in the same folder

# 2. Select features (inputs) and target (output)
X = iris_df.drop(columns=["Id", "Species"])   # Features
y = iris_df["Species"]                        # Target labels

# 3. Encode the target labels (convert species names to numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on the test data
y_pred = model.predict(X_test)

# 7. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
