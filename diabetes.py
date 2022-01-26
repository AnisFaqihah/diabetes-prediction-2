#import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

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

#Show 7 rows of dataset 
st.markdown('This is the sample dataset.')
st.write(df.head(7)) 

#Button to make prediction
if st.button('Make prediction'):
    #some preprocessing steps
    dataset = pd.get_dummies(df, columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'BMI', 'DPT', 'age'])
    model(dataset)
 
#Split the data into independent 'X' and dependent 'Y' variables
def model(df):
   Y = dataset['outcome']
   X = dataset.drop(['outcome'], axis = 1)
   # Data splitting
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test) 

#Get input from users 
st.sidebar.header('User Input Section')    
 
#Get the feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('Number of time pregnant', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose concentration', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Diastolic blood pressure (mm Hg)', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('2 hour serum insulin (mu U/ml)', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('Body Mass Index/BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    #Store dictionary into variable
    user_data = {'Pregnancies' : pregnancies,
                 'Glucose' : glucose,
                 'Blood pressure' : blood_pressure,
                 'Skin thickness' : skin_thickness,
                 'Insulin' : insulin,
                 'BMI' : BMI,
                 'DPF' : DPF,
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

st.sidebar.header('Fine tuning Section via HyperParameter Optimization') 
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 50, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators for Random Forest (n_estimators)', 0, 500, (10,50), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)

st.sidebar.write('---')
parameter_max_features =st.sidebar.multiselect('Max Features (You can select multiple options)',['auto', 'sqrt', 'log2'],['auto'])
parameter_max_depth = st.sidebar.slider('Maximum depth', 5, 15, (5,8), 2)
parameter_max_depth_step=st.sidebar.number_input('Step size for max depht',1,3)

st.sidebar.write('---')
parameter_criterion = st.sidebar.selectbox('criterion',('gini', 'entropy'))
st.sidebar.write('---')
parameter_cross_validation=st.sidebar.slider('Number of Cross validation split', 2, 10)

st.sidebar.subheader('Other Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
"""
if parameter_n_estimators[0] is 5 and parameter_n_estimators[1] 25 and parameter_n_estimators_step is 5
then array will be [5,10,15,20,25]
"""
max_depth_range =np.arange(parameter_max_depth[0],parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
param_grid = dict(max_features=parameter_max_features,
n_estimators=n_estimators_range,max_depth=max_depth_range)

#Create and train the model
rf = RandomForestClassifier(random_state=parameter_random_state,
         bootstrap=parameter_bootstrap,
         n_jobs=parameter_n_jobs)

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=parameter_cross_validation)
grid.fit(X_train, Y_train)

st.subheader('Model Performance')
Y_pred_test = grid.predict(X_test)
    
st.write('Accuracy score of given model')
st.info( accuracy_score(Y_test, Y_pred_test) )   

st.write("The best parameters are %s with a score of %0.2f" %(grid.best_params_, grid.best_score_))

st.subheader(‘Model Parameters’)
st.write(grid.get_params())

#Newline
st.markdown("***")

#Open and display image
image3 = Image.open('diabetes3.png')
st.image(image3, use_column_width=True)
