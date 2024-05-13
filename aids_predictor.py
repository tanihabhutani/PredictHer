# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace 'file_path.csv' with the path to your dataset)
file_path = 'hiv1_data.csv'
data = pd.read_csv(file_path)

# Explore the dataset
print(data.head())

# Split the dataset into features (X) and target variable (y)
X = data.drop('cleave', axis=1)  # Features
y = data['cleave']  # Target variable (1 for cleavage, 0 for non-cleavage)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))