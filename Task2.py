# CODSOFT - Task 2: Movie Rating Prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Genre_Action': [1,0,1,0,1],
    'Director_Score': [8,7,9,6,8],
    'Budget': [100,50,120,30,80],
    'Rating': [8.2,6.5,8.8,5.9,7.8]
}
df = pd.DataFrame(data)
X = df[['Genre_Action','Director_Score','Budget']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("Task 2 - Movie Rating Prediction Completed")
