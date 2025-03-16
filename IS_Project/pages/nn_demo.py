import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, permutation_test_score
import seaborn as sns
import pickle as p
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
# %matplotlib inline

model = load_model("models/neuron_model.keras")
pclass = st.number_input("Pclass", min_value=1, max_value=3, value=1)
sex = st.selectbox("Sex", ["male", "female"])
fare = st.number_input("Fare", min_value=0.0, value=50.0)

# Encode 'Sex'
le_sex = LabelEncoder()
le_sex.fit(["male", "female"])
sex_encoded = le_sex.transform([sex])[0]

# Combine input data
input_data = pd.DataFrame([[pclass, sex_encoded, fare]], columns=["Pclass", "Sex_n", "Fare"])

# Make prediction
prediction = model.predict(input_data)[0]

# Display the result
st.subheader("Prediction")
if st.button("🔮 Predict"):
    if prediction > 0.5 :
        st.write("Survived")
    else:
        st.write("Did not survived")

