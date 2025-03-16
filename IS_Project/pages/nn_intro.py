from ydata_profiling import ProfileReport
#Imports  ProfileReport class จาก ydata_profiling library

# Commented out IPython magic to ensure Python compatibility.
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
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.linear_model import LogisticRegression
# %matplotlib inline

st.markdown("# neuronnetworks 🎈")
st.sidebar.markdown("# neuronnetworks 🎈")

st.title("🔍 Neuron Networks Intro")
st.write(""" 
         ### Why I Chose the Titanic Dataset for My Neural Network Model

        Ever since I first learned about machine learning, I was fascinated by how AI could be used to make predictions, uncover hidden patterns, and even save lives. 
         But one question kept haunting me: Could AI have saved more people on the Titanic? With that thought in mind, I set sail on my machine-learning journey. 
         The Titanic dataset, provided by [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file), contained real passenger data
        """
)

        
st.write("### Data Preparation")
st.write(""" The dataset is read from a CSV file (heart.csv), Loads the dataset into a Pandas DataFrame. 
         This dataset contains information about Titanic passengers, such as their class, sex, fare, and whether they survived.
         
         df = pd.read_csv("tested.csv")
 
         

         """ )

from PIL import Image as img
st.image(img.open("images/nn/df_describe1.png"))
st.write("""Summarizing datasets:
        
        df.describe()
       
        """)
st.image(img.open("images/nn/df_isnull.png"))
st.write("""Checking Missing Values:
        
        df.isnull().sum()
          
         """)
st.image(img.open("images/nn/featured1.png"))
st.write("""Pick only Important Data and make new DataFrame:featured
         
        featured = pd.DataFrame({'Pclass':df['Pclass'] , 'Sex':df['Sex'] , 'Fare':df['Fare'] , 'Survived':df['Survived'] })

        featured
        
          
        
         """)
st.write("""Delete NaN rows
         
         
         
         
         
         
         
         """)




st.write("""   
         
         
         featured=featured.dropna()  
         
         
         
         
         """)

st.image(img.open("images/nn/df_nan1.png"))
st.image(img.open("images/nn/y.png"))
st.write("""Make DataFrame y to be test label and Make DataFrame x to be train label""")
st.code((""" 
         y = featured['Survived']
x = featured.drop(columns = ['Survived'],axis='columns')
         """),language="python") 

st.write(""" from sklearn.preprocessing import LabelEncoder
            
         le_sex=LabelEncoder()
            
         x['Sex_n']= le_sex.fit_transform(x['Sex'])
            
         x
            
         x_new=x.drop('Sex', axis='columns')
            
         x_new
         
         
         """)
st.image(img.open("images/nn/minussex.png"))
st.write("""Converts the categorical "Sex" column into numerical values using Label Encoding (Male = 0, Female = 1).
Drops the original text-based "Sex" column and replaces it with the new numeric version.
         """)

st.write("""### Model Used
         
    from sklearn.model_selection import train_test_split
    x_train, x_test,y_train,y_test= train_test_split(x_new,y,test_size=0.30, random_state=42)
    x_train
         
   

          
         """)
st.write("""My model is a feedforward artificial neural network (ANN) 
         designed for binary classification—predicting whether a Titanic passenger survived (1) or not (0). 
         It is built using Keras with the TensorFlow backend.""")
       




st.image(img.open("images/nn/model_summary.png"))
st.image(img.open("images/nn/result.png"))




st.write(""" ### 📈Correlation Heatmap:
         from sklearn.metrics import confusion_matrix
import seaborn as sns
# Get predicted labels
y_pred = model.predict(x_test)
# Convert predicted probabilities to class labels (0 or 1)
y_pred_classes = (y_pred > 0.5).astype(int)
# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()      """)
st.image(img.open("images/nn/corr.png"))
st.write("""Converts probabilities into class labels (0 or 1).
Generates a confusion matrix to analyze true positives, false positives, etc.
Uses seaborn’s heatmap to visualize results..""")
st.page_link("pages/nn_demo.py", label="try demo")