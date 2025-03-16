# -*- coding: utf-8 -*-
"""Assignment5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1zWtezjXp2mxGF0c7IZReqkIPV-HTmOEj

**Thanathorn Ruengtorwong 6604062630226**

#  Assignment 1
"""


#Installs the ydata-profiling package.
from ydata_profiling import ProfileReport
#Imports  ProfileReport class จาก ydata_profiling library

"""Installs the ydata-profiling package"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
# %matplotlib inline
#โค้ด Python สำหรับการนำเข้าไลบรารีและเครื่องมือที่ใช้ในงาน AI โดยเฉพาะ Machine Learning และ Deep Learning

# from google.colab import files
# uploaded = files.upload()
# #สร้างที่อัพโหลดไฟล์จากคอมเข้าสู่ google colab

df = pd.read_csv("tested.csv")
#โค้ดอ่านไฟล์ข้อมูล CSV

profile = ProfileReport(
    df, title="tested", html={"style": {"full_width": True}} , sort=None
    )
#คำสั่งฟังก์ชันจากไลบรารี ydata-profiling เพื่อสร้าง EDA ของ DataFrame df โดยอัตโนมัติ

profile.to_notebook_iframe()
#แสดงรายงานที่สร้างโดย ProfileReport ในรูปแบบ iframe ภายใน Jupyter Notebook หรือ Google Colab โดยตรง

df.shape
#เป็นคำสั่งในไลบรารี pandas ที่ใช้เพื่อดู ขนาดของ DataFrame df ซึ่งจะแสดงจำนวนแถวและจำนวนคอลัมน์

df.columns
#เป็นคำสั่งใน pandas ที่ใช้สำหรับดูรายชื่อ คอลัมน์ ทั้งหมดใน DataFrame df

df.isnull().any()
#สร้าง DataFrame ใหม่ที่แสดงว่าแต่ละค่าของ df เป็น NaN  หรือไม่

df.describe(include = 'all')
#แสดงข้อมูลสรุปของ DataFrame df สำหรับ ทุกประเภทข้อมูล ไม่ว่าจะเป็น ตัวเลข หรือ ประเภทอื่น ๆ (เช่น ข้อความ)

featured = pd.DataFrame({'Pclass':df['Pclass'] , 'Sex':df['Sex'] , 'Fare':df['Fare'] , 'Survived':df['Survived'] })

featured

featured = featured.dropna()
#ลบค่า NaN ใน DataFrame featured

featured
#แสดงข้อมูลที่เปลี่ยนแปลงตามข้างต้น

y = featured['Survived']
x = featured.drop(columns = ['Survived'],axis='columns')
#เก็บข้อมูลของคอลัมน์ Survived ไว้ใน DataFrame y และ drop คอลัมน์ Survived จาก Dataframe featured และเก็บไว้ในตัวแปร DataFrame x

x.head()
#แสดง df ที่เก็บไว้ใน DataFrame x

y.value_counts()
#นับข้อมูลทั้งหมดใน DataFrame y

from sklearn.preprocessing import LabelEncoder
le_sex=LabelEncoder()
x['Sex_n']= le_sex.fit_transform(x['Sex'])
x

x_new=x.drop('Sex', axis='columns')
x_new

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test= train_test_split(x_new,y,test_size=0.30, random_state=42)
x_train
#แบ่งข้อมูลออกเป็น ชุดฝึก training set และ test set ด้วยฟังก์ชัน train_test_split() จากไลบรารี sklearn.model_selection

"""# Assignment 5"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers, callbacks
from tensorflow.keras.optimizers import AdamW
from keras.losses import mean_squared_error
#design architecture of model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
    input_shape=(3,)),


    keras.layers.Dense(64, activation='relu'),


    keras.layers.Dense(32, activation='relu'),


    keras.layers.Dense(16, activation='relu'),


    keras.layers.Dense(1, activation='sigmoid')  # Output layer




])
model.summary()

#compile model
model.compile(loss= 'binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1.0),
              metrics=['accuracy'])



#train model
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[lr_scheduler,early_stopping])

model.evaluate(x_test, y_test)

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
plt.show()

#create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
#fit the model
history = model.fit(x_train, y_train, epochs=100, callbacks=[lr_scheduler])
#plot
lrs = 1e-4 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs. Loss")


model.save('models/neuron_model.keras')