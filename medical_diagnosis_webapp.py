# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg
import pickle

image1 = mpimg.imread('h1.png')     
image2 = mpimg.imread('h2.png')     
image3 = mpimg.imread('h3.png')     
image4 = mpimg.imread('h4.png')     
image5 = mpimg.imread('h5.png')     
image6 = mpimg.imread('h6.png')     
image7 = mpimg.imread('h7.png')     
image8 = mpimg.imread('h8.png')     
image9 = mpimg.imread('h9.png')     
image10 = mpimg.imread('h10.png')     
image11 = mpimg.imread('h11.png')     
image12 = mpimg.imread('h12.png')     
image13 = mpimg.imread('h13.png')     
image14 = mpimg.imread('h14.png')     
image15 = mpimg.imread('h15.png')     
image16 = mpimg.imread('h16.png')     
image17 = mpimg.imread('h17.png')     
image18 = mpimg.imread('h18.png')     
image19 = mpimg.imread('h19.png')
image20 = mpimg.imread('a1.png')


st.set_page_config(page_title='Medical Diagnosis System',layout='wide')






Selection = st.sidebar.selectbox("Select Option", ("Home","Heart Disease Detection","Diabetes Detection","Parkinson's Disease Detection","Exploratory Data Analysis"))

if Selection == "Home":
  

    st.write("""# Medical Diagnosis WebApp

            """)

    st.image(image20)
    st.write("""
        **Multiple Disease Prediction & Analysis using Machine Learning (Heart Disease, Diabetes & Parkinson's Disease)** 

        """)



    st.write("""
            App Link: https://share.streamlit.io/shubhamkg1999/medical-diagnosis-webapp-aiactr/main/medical_diagnosis_webapp.py


            """)
    st.write("""
            
            Repository: https://github.com/ShubhamKG1999/Medical-Diagnosis-WebApp-AIACTR


                """)
    st.write("""
        Many of the existing machine learning models for health care analysis are concentrating on one
disease per analysis. Like one model for diabetes analysis, one for heart disease analysis, one for Parkinson's disease like that. There is no common system where one analysis can perform more than one
disease prediction. In this project, we propose a system which is used to predict multiple diseases
such as Heart Disease, Diabetes & Parkinson's disease by using the Streamlit framework along with Localtunnel.

        """)

st.write("""

            **© Shubham Kumar Gupta, Ankit Sharma, & Yatharth Singh (2022)** 


                """)

if Selection == "Heart Disease Detection":

    data = pd.read_csv('heart.csv')
    X = data[['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    y = data['target']

    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))

    st.title("1.1. Heart Disease Detection")
    st.write(data.head(5))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
    age = st.sidebar.slider('age',29,77,40,1)
    cp = st.sidebar.slider('cp', 0, 3, 1, 1)
    sex = st.sidebar.slider('sex',0,1,0,1)
    trestbps = st.sidebar.slider('trestbps', 94, 200, 80, 1)
    chol = st.sidebar.slider('chol', 126, 564, 246, 2)
    fbs = st.sidebar.slider('fbs', 0, 1, 0, 1)
    restecg = st.sidebar.slider('restecg', 0, 2, 1, 1)
    exang = st.sidebar.slider('exang', 0, 1, 0, 1)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 3.2, 0.2)
    slope= st.sidebar.slider('slope', 0, 2, 1, 1)
    ca= st.sidebar.slider('ca', 0, 4, 2, 1)
    thal= st.sidebar.slider('thal', 0, 3, 1, 1)
    thalach = st.sidebar.slider('thalach',71,202,150,1)
    
    X_test_sc = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)
    
    load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

# Apply model to make predictions
    prediction = load_clf.predict(X_test_sc)
    prediction_proba = load_clf.predict_proba(X_test_sc)
    
    answer = prediction[0]
        
    if answer == 0:

        st.title("**The prediction is that the Heart Disease was not Detected**")
   
    else:   
        st.title("**The prediction is that the Heart Disease was Detected**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')

elif Selection == "Exploratory Data Analysis":

    st.title("1.4 Exploratory Data Analysis")
    st.write("'Age and Target' Countplot")
    st.image(image19)

    st.write("'Chest Pain' Countplot")
    st.image(image3)

    st.write("'Sex' Countplot")
    st.image(image2)

    st.write("'trestbps' Distplot")
    st.image(image4)

    st.write("'trestbps' Histogram")
    st.image(image5)

    st.write("'chol' Distplot")
    st.image(image6)

    st.write("'chol' Histogram")
    st.image(image7)

    st.write("'fbs' Countplot")
    st.image(image8)

    st.write("'restecg' Countplot")
    st.image(image9)

    st.write("'thalach' Distplot")
    st.image(image10)

    st.write("'thalach' Histogram")
    st.image(image11)

    st.write("'exang' Countplot")
    st.image(image12)

    st.write("'oldpeak' Distplot")
    st.image(image13)

    st.write("'oldpeak' Histogram")
    st.image(image14)

    st.write("'oldpeak' Histogram")
    st.image(image15)

    st.write("'slope' Countplot")
    st.image(image16)

    st.write("'ca' Countplot")
    st.image(image17)

    st.write("'thal' Countplot")
    st.image(image18)

elif Selection == "Diabetes Detection":

    data = pd.read_csv('diabetes.csv')
    X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
              'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = data['Outcome']

    # st.markdown('The Diabetes dataset used for training the model is:')
    # st.write(data.head(5))

    st.title("1.2 Diabetes Detection")
    st.write(data.head(5))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # st.sidebar.header('2. Set Parameters'):
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3, 1)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 120, 1)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 70, 2)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 20, 3)
    Insulin = st.sidebar.slider('Insulin', 0, 846, 79, 6)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 20.0, 1.1)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.420, 0.470, 0.004)
    Age = st.sidebar.slider('Age', 21, 81, 33, 1)


    X_test_sc = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

    # logregs = LogisticRegression()
    # logregs.fit(X_train, y_train)
    # y_pred_st = logregs.predict(X_test_sc)

    load_clf = pickle.load(open('diabetes_clf.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(X_test_sc)
    prediction_proba = load_clf.predict_proba(X_test_sc)

    answer = prediction[0]

    if answer == 0:

        st.title("**The prediction is that the Diabetes was not Detected**")

    else:
        st.title("**The prediction is that the Diabetes was Detected**")

    st.write('Note: This prediction is based on the Machine Learning Algorithm, Random Forrest.')

elif Selection == "Parkinson's Disease Detection":

    st.write("**1.3 Parkinson's Disease Detection**")
    data = pd.read_csv('parkinsons.csv')
    st.write(data.head(5))
    parkinsons_model = pickle.load(
        open('parkinsons.pkl', 'rb'))

    # page title
    st.title("1.3 Parkinson's Disease Detection")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

st.sidebar.title("Created By:")
st.sidebar.subheader("Shubham Kumar Gupta")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/shubham-kumar-gupta-1720751a7/)")
st.sidebar.subheader("Ankit Sharma")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/ankit-sharma-73234b19a/)")
st.sidebar.subheader("Yatharth Singh")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/yatharth-singh-437ab2171/)")
st.sidebar.subheader("")
st.sidebar.subheader("[GitHub Repository](https://github.com/ShubhamKG1999/Heart-Disease-Detection-App-AIACTR)")