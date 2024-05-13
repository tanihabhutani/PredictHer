# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv("depression_anxiety.csv")

# Split data into features and target variable(s)
X = data.drop(columns=["Depression_Label", "Anxiety_Label"])  # Assuming labels are named "Depression_Label" and "Anxiety_Label"
y_depression = data["Depression_Label"]
y_anxiety = data["Anxiety_Label"]

# Split data into training and testing sets
X_train, X_test, y_train_depression, y_test_depression = train_test_split(X, y_depression, test_size=0.2, random_state=42)
X_train, X_test, y_train_anxiety, y_test_anxiety = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier for depression
rf_depression = RandomForestClassifier(n_estimators=100, random_state=42)
rf_depression.fit(X_train, y_train_depression)

# Make predictions for depression
predictions_depression = rf_depression.predict(X_test)

# Evaluate depression model
print("Depression Model Accuracy:", accuracy_score(y_test_depression, predictions_depression))
print("Depression Model Classification Report:")
print(classification_report(y_test_depression, predictions_depression))

# Initialize and train the Random Forest classifier for anxiety
rf_anxiety = RandomForestClassifier(n_estimators=100, random_state=42)
rf_anxiety.fit(X_train, y_train_anxiety)

# Make predictions for anxiety
predictions_anxiety = rf_anxiety.predict(X_test)

# Evaluate anxiety model
print("Anxiety Model Accuracy:", accuracy_score(y_test_anxiety, predictions_anxiety))
print("Anxiety Model Classification Report:")
print(classification_report(y_test_anxiety, predictions_anxiety))