import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# 1. DATA LOADING
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# 2. DATA CLEANING & FEATURE ENGINEERING
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
# 3. SPLITTING DATA
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 4. MODEL TRAINING
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 5. EVALUATION
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Titanic Model Accuracy: {accuracy * 100:.2f}%")