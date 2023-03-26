# from ast import If
# from turtle import width
# from unicodedata import name
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from PIL import Image
from sklearn.metrics import accuracy_score

st.title("DIABETES PREDICTION")

df = pd.read_csv("diabetes_csv_dataset.csv") 
# pd.read_csv
def validation (name):
    
        for char in name:
            if  not (("A" <= char and char <= "Z") or ("a" <= char and char <= "z") or (char == " ")):
                return False
        return True

tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs(["𝐇𝐎𝐌𝐄","        ", "        ","        ", "𝐂𝐇𝐄𝐂𝐊𝐔𝐏","        "])
with tab1:
    st.write("")
    st.title('WELCOME TO OUR DIABETES PAGE')
    image = Image.open("DIABETES.jpg")
    st.write("")
    st.image(image,width=700)

    st.write('')
    
    st.header('𝐒𝐘𝐌𝐏𝐓𝐎𝐌𝐒 𝐀𝐋𝐄𝐑𝐓 :')
    st.write('')
    video_file = open('home_page_video.mp4','rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
    
with tab5:

    st.subheader("𝗘𝗡𝗧𝗘𝗥 𝗬𝗢𝗨𝗥 𝗡𝗔𝗠𝗘 :")
    st.write('')
    st.info('𝐍𝐀𝐌𝐄 𝐈𝐍 𝐀𝐀𝐃𝐇𝐀𝐀𝐑 𝐂𝐀𝐑𝐃')
    name_inp = st.text_input('','')
    result = validation(name_inp)
    if result == True:
        name_input = name_inp
    else:
        st.warning('𝐏𝐥𝐞𝐚𝐬𝐞 𝐞𝐧𝐭𝐞𝐫 𝐚 𝐯𝐚𝐥𝐢𝐝 𝐧𝐚𝐦𝐞..!!')


    st.write("")
    st.subheader("𝗣𝗥𝗘𝗚𝗡𝗔𝗡𝗖𝗬 :")
    preg = st.slider('', 0, 20, 0)

    st.write("")
    st.subheader("𝗚𝗟𝗨𝗖𝗢𝗦𝗘 :")
    glu = st.slider('', 0, 500, 0)

    st.write("")
    st.subheader("𝗕𝗟𝗢𝗢𝗗 𝗣𝗥𝗘𝗦𝗦𝗨𝗥𝗘 :")
    bp = st.slider('', 0, 200, 0)

    st.write("")
    st.subheader("𝗦𝗞𝗜𝗡 𝗧𝗛𝗜𝗖𝗞𝗡𝗘𝗦𝗦 :")
    skin = st.slider('', 0, 100, 0)

    st.write("")
    st.subheader("𝗜𝗡𝗦𝗨𝗟𝗜𝗡 :")
    insulin = st.slider('', 0, 1000, 0)

    
    
    st.subheader("𝗕𝗠𝗜 :")
    bmi = st.number_input('𝐄𝐍𝐓𝐄𝐑 𝐘𝐎𝐔𝐑 𝐁𝐌𝐈 𝐕𝐀𝐋𝐔𝐄')

    st.write("")
    st.subheader("𝗗𝗜𝗔𝗕𝗘𝗧𝗘𝗦 𝗣𝗘𝗗𝗜𝗚𝗥𝗘𝗘 𝗙𝗨𝗡𝗖𝗧𝗜𝗢𝗡 :")
    dpf = st.number_input('𝐄𝐍𝐓𝐄𝐑 𝐘𝐎𝐔𝐑 𝐃𝐏𝐅 𝐕𝐀𝐋𝐔𝐄')

    st.write("")
    st.subheader("𝗔𝗚𝗘 :")
    a_g_e = st.number_input('𝐄𝐍𝐓𝐄𝐑 𝐘𝐎𝐔𝐑 𝐀𝐆𝐄')
    st.write('')

    


    scaler = StandardScaler()
    X = df.drop(columns = 'Outcome', axis=1)
    scaler.fit(X)
    standardized_data = scaler.transform(X)
    X = standardized_data
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    if st.button("𝐑𝐄𝐒𝐔𝐋𝐓"):

        input_data = (preg,glu,bp,skin,insulin,bmi,dpf,a_g_e)

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        # standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        

        prediction = classifier.predict(std_data)
        

        if (prediction[0] == 0):
            st.write('')
            st.success("{} IS NOT AFFECTED WITH DIABETES".format(name_input))
            
        else:
            st.write('')
            st.success("{} IS DIABETIC".format(name_input))
            st.write('')
            st.header('𝐇𝐄𝐑𝐄 𝐀𝐑𝐄 𝐓𝐇𝐄 𝐑𝐄𝐌𝐄𝐃𝐈𝐄𝐒..!!\n 𝐏𝐋𝐄𝐀𝐒𝐄 𝐖𝐀𝐓𝐂𝐇 𝐈𝐓')
            st.write('')
            video_file = open('result_video.mp4','rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
