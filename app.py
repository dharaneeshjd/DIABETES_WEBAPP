# from ast import If
# from turtle import width
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from PIL import Image
from sklearn.metrics import accuracy_score

st.title("DIABETES PREDICTION")

df = pd.read_csv("C:\python\streamlit_diabetes\diabetes_csv_dataset.csv") 
# pd.read_csv


nav = st.sidebar.radio("OPTIONS",["HOME","PREVIEW","CONTRIBUTE"])
if nav == "HOME":
    
    image = Image.open('C:\python\streamlit_diabetes\DIABETES.jpg')
    st.write("")
    st.image(image,width=700)
    
if nav == "PREVIEW":
    if st.checkbox("PREVIEW DATASET"):
        data=df
        if st.button("HEAD"):
            st.write(df.head())
        if st.button("SHAPE"):
            st.write(df.shape)
        if st.button("DESCRIBE"):
            st.write(df.describe())
        if st.button("COUNTS"):
            st.write(df['Outcome'].value_counts())
        if st.button("GROUPBY_MEAN"):
            st.write(df.groupby('Outcome').mean())
            st.write("fix")
        if st.button("DROP_OUTCOME"):
            df.groupby('Outcome').mean()
            X= df.drop(columns = 'Outcome', axis=1)
            st.write(X)
        if st.button("OUTCOME"):
            re = df.groupby('Outcome').mean()
            Y = df['Outcome']
            n = np.array(Y)
            # st.write(n.shape)
            nd=n.reshape([2,8])
            st.write(nd)
        if st.button("STD DATA"):
            scaler = StandardScaler()
            X = df.drop(columns = 'Outcome', axis=1)
            scaler.fit(X)
            standardized_data = scaler.transform(X)
            st.write(standardized_data)
        
        scaler = StandardScaler()
        X = df.drop(columns = 'Outcome', axis=1)
        scaler.fit(X)
        standardized_data = scaler.transform(X)
        X = standardized_data
        Y = df['Outcome']
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
        sh = st.radio("SELECT THE SHAPE",["X-SHAPE","X TRAIN SHAPE","X TEST SHAPE"])
        if sh == "X-SHAPE":
            st.write(X.shape)
        if sh=="X TRAIN SHAPE":
            st.write(X_train.shape)
        if sh=="X TEST SHAPE":
            st.write(X_test.shape)
    scaler = StandardScaler()
    X = df.drop(columns = 'Outcome', axis=1)
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    if st.button("ACCURACY OF TRAIN DATA"):
        X_train_prediction = classifier.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
        st.write('Accuracy score of the training data : ', training_data_accuracy)
    if st.button("ACCURACY OF TEST DATA"):
        X_test_prediction = classifier.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        st.write('Accuracy score of the test data : ', test_data_accuracy)
if nav == "CONTRIBUTE":


    # st.write("")
    # st.subheader("SURNAME:")
    # sr = st.text_input('','Please enter here...!')


    st.write("")
    st.subheader("ENTER YOUR NAME:")
    name_input = st.text_input('','Please enter here...!')

    


    st.write("")
    st.subheader("PREGNANCY:")
    preg = st.slider('', 0, 20, 0)

    st.write("")
    st.subheader("GLUCOSE:")
    glu = st.slider('', 0, 500, 0)

    st.write("")
    st.subheader("BLOOD PRESSURE:")
    bp = st.slider('', 0, 200, 0)

    st.write("")
    st.subheader("SKIN THICKNESS:")
    skin = st.slider('', 0, 100, 0)

    st.write("")
    st.subheader("INSULIN:")
    insulin = st.slider('', 0, 1000, 0)

    
    st.write("")
    st.subheader("BMI:")
    bmi = st.number_input('ENTER YOUR BMI VALUE')

    st.write("")
    st.subheader("DIABETES PEDIGREE FUNCTION:")
    dpf = st.number_input('ENTER YOUR DPF VALUE')

    st.write("")
    st.subheader("AGE:")
    a_g_e = st.number_input('ENTER YOUR AGE')

    


    scaler = StandardScaler()
    X = df.drop(columns = 'Outcome', axis=1)
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    if st.button("   RESULT   "):

        input_data = (preg,glu,bp,skin,insulin,bmi,dpf,a_g_e)

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        

        prediction = classifier.predict(std_data)
        

        if (prediction[0] == 0):
            st.write(name_input,"IS NOT AFFECTED WITH DIABETES")
        else:
            st.write(name_input,"IS DIABETIC")




    
    
    







    
