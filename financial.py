import pandas as pd
import streamlit as st
import numpy as np
import joblib

st.header('FINANCIAL INCLUSION IN EAST AFRICA')

st.image("./openinaccount.jpg")
st.write('The dataset contains demographic information and what financial services are used by approximately individuals across  a few countries in East Africa. The ML model role is to predict which individuals are most likely to have or use a bank account.')


education = ['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA']

with st.sidebar:
    country= st.selectbox("Country",["Kenya", "Rwanda", "Tanzania", "Uganda"])
    location_type= st.selectbox("Where do you live", ["Rural", "Urban"])
    cellphone_access=st.selectbox("Do you own a cell phone",["Yes", "No"])
    age_of_respondent= st.number_input("How old are you?",18,100)
    gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married/Living together", "Widowed", "Single/Never Married","Divorced/Seperated", "Dont know"])
    education_level=st.selectbox('Education Level',[x for x in education] )
  

userinput ={'country': country,
            'location_type': location_type,
            'cellphone_access':cellphone_access,
            'age_of_respondent':age_of_respondent, 
            'gender_of_respondent': gender_of_respondent, 
            'marital_status': marital_status,
            'education_level': education_level, 
          
            }



if st.button('predict'):
    userdata = pd.DataFrame([userinput])

    
    country_encoder = joblib.load('joblib_CountryEncoder.sav')
    userdata['country']= country_encoder.transform(userdata['country'])

    marital_encoder = joblib.load('joblib_MaritalEncoder.sav')
    userdata['marital_status'] = marital_encoder.transform(userdata['marital_status'])

    cellphone_encoder = joblib.load('joblib_cellphone_encoder.sav')
    userdata['cellphone_access'] = cellphone_encoder.transform(userdata['cellphone_access'])

    location_encoder =joblib.load('joblib_locationEncoder.sav')
    userdata['location_type'] = location_encoder.transform(userdata['location_type'])

    genderencoder = joblib.load('joblib_genderencoder.sav')
    userdata['gender_of_respondent']= genderencoder.transform(userdata['gender_of_respondent'])

    education_encoder = joblib.load('joblib_educationencoder.sav')
    userdata['education_level'] = education_encoder.transform(userdata['education_level'])

    model = joblib.load('joblib_clf.sav')
    prediction = model.predict(userdata)

    if prediction == 0:
        st.write('Is likely to have or use a Bank account')
    else:
        st.write('Will not have or use a Bank account')


   



