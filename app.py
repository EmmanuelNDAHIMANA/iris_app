# app.py
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.title("Iris Classification (Quick Demo)")

# Load data
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
species = iris.target_names

# Train a small model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Inputs for a single flower
st.sidebar.header("Input features")
sl = st.sidebar.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()), 5.1)
sw = st.sidebar.slider("Sepal width (cm)",  float(X["sepal width (cm)"].min()),  float(X["sepal width (cm)"].max()), 3.5)
pl = st.sidebar.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()), 1.4)
pw = st.sidebar.slider("Petal width (cm)",  float(X["petal width (cm)"].min()),  float(X["petal width (cm)"].max()), 0.2)

sample = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]

st.subheader("Prediction")
st.write(f"**Species:** {species[pred]}")
st.write(pd.DataFrame({"species": species, "probability": proba}).sort_values(
    by="probability",
    ascending=False
)
)
