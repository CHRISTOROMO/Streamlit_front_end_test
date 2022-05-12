from random import random
from turtle import color
import streamlit as st
import _tkinter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.markdown(
    """
    <style>
    .main {
        background-color: #6b8829;
        background-image: url("https://ak.picdn.net/shutterstock/videos/14331430/thumb/1.jpg");
        }
        </style>
        """,
        unsafe_allow_html=True,
)


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache
def get_data(filename):
    wine = pd.read_csv(filename)
    
    return wine

with header:
    st.title('Welcome my Data Science Web App')
    st.text('This is a web app that will help you to learn about the data')
   
with dataset:
    st.header('Predicting Wine Quality')
    st.text('This is a dataset that contains the following information: ')
    
    wine = get_data('winequality-red.csv')
    st.write(wine.head())
    
    st.subheader('Wine Data for Alcohol')
    alcohol = pd.DataFrame(wine['alcohol'].value_counts())
    st.bar_chart(alcohol)
    
    st.subheader('Wine Data for pH')
    pH = pd.DataFrame(wine['pH'].value_counts()).head(20)
    st.line_chart(pH)
    
with features:
    st.header('Features')
    
    st.markdown('* **First Feature:** I created this feature to enable ...')
    st.markdown('* **Second Feature:** I created this feature to enable ...')


with model_training:     
    st.header('Time to model!')
    st.text('This is a model that will help you to predict the outcome of the hospital')
    
    sel_col = st.container()
    disp_col = st.container()
    
    max_depth = sel_col.slider('What should be the max_depth of the model', min_value=10,max_value=100,value=20,step=10)
    
    n_estimators = sel_col.selectbox('How many estimators? ', options=[100,200,300,'No Limit'], index=0) 
    
    if n_estimators == 'No Limit':
        cls = RandomForestClassifier(max_depth=max_depth)
    else:
        cls = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
  
        
    input_feature = sel_col.text_input('Which feature should be used as input? ', 'alcohol')
    
    sel_col.text('Here is the list of Features: ')
    sel_col.write(wine.columns) 
    
    X = wine[input_feature].values.reshape(-1,1)
    y = wine['quality']
    
    cls.fit(X, y)
    y_pred = cls.predict(X)    
    
    disp_col.subheader('Accuracy of the model: ')
    disp_col.write(accuracy_score(y, y_pred))
    
    
    