import pandas as pd
from sklearn.linear_model import
LogisticRegression
from sklearn.model_selection import
train_test_split
import joblib

df = pd.read_csv("data/iris.csv")
X = df.drop("target",azis=1)
y = df["target"]

X_train, X_test, y_train,y_test=
train_test_split(X,y)

joblib.dump(model, "model.model/joblib")
