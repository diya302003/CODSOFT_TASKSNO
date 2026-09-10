# CODSOFT - Task 1: Titanic Survival Prediction
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'embarked'])

# Convert categorical to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features
X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
print("Task 1 Completed Successfully")
