import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Clean column names
data.columns = data.columns.str.strip()

# 🔴 IMPORTANT FIX: Remove unwanted columns
data = data.drop(["Student_ID", "Salary_Offered_USD"], axis=1)

# Convert categorical to numeric
data["Placement_Offer"] = data["Placement_Offer"].map({"Yes": 1, "No": 0})
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

data["Degree"] = data["Degree"].map({
    "Engineering": 0,
    "Computer Science": 1,
    "Data Science": 2,
    "Business": 3,
    "Arts": 4
})

# Split features and target
X = data.drop("Placement_Offer", axis=1)
y = data["Placement_Offer"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained successfully!")