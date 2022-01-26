#import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create title and sub-title
st.write("""
# Diabetes Prediction
Predict if someone has diabetes using Random Forest!
""")

#Open and display image
image1 = Image.open('diabetes4.png')
st.image(image1, use_column_width=True)

#Get the data
path = "https://github.com/AnisFaqihah/diabetes-prediction-2/raw/main/diabetes.csv"
df = pd.read_csv(path)

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

#Split the dataset into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

#Get the feature input from user
def get_user_input():
    pregnancies = st.sidebar.number_input('Number of time pregnant',0,17)
    glucose = st.sidebar.slider('Glucose concentration', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Diastolic blood pressure (mm Hg)', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('2 hour serum insulin (mu U/ml)', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('Body Mass Index/BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.number_input('Age',21,81)
   
    #Store dictionary into variable
    user_data = {'Number of time pregnant' : pregnancies,
                 'Glucose concentration' : glucose,
                 'Diastolic blood pressure' : blood_pressure,
                 'Skin thickness' : skin_thickness,
                 '2 hour serum insulin' : insulin,
                 'Body Mass Index' : BMI,
                 'Diabetes Predigree Function' : DPF,
                 'Age' : age
                }
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User Input: ')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test)) * 100)+'%')

#Store the models predictions in variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader and display the prediction
st.subheader('Prediction: ')
st.write(prediction)

#Newline
st.markdown("*")

#Open and display image
image3 = Image.open('diabetes3.png')
st.image(image3, use_column_width=True)
