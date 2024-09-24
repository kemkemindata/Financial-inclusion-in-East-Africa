## Financial Inclusion Prediction in East Africa

## INTRODUCTION
This project uses a machine learning model to predict whether individuals in East Africa are likely to have or use a bank account based on their demographic information. 
The project is built using Streamlit for the user interface and Scikit-learn for the machine learning components.
This application allows users to input personal and demographic information to predict the likelihood of financial inclusion (i.e., having or using a bank account). 
The model is trained on data from several East African countries: Kenya, Rwanda, Tanzania, and Uganda. 
The prediction is based on multiple factors including age, gender, education, and location.


Table of Contents

-INTRODUCTION

-FEATURES

-INSTALLATION

-RUN

-DATASET

-MODEL DESCRIPTION

-FILE STRUCTURE



## FEATURES

-User Input: A sidebar allows users to input data such as age, country, education, and marital status.

-Prediction: The application uses a trained machine learning model to predict whether the user is likely to have or use a bank account.

-Visualization: Displays relevant images and text output based on the prediction.

## INSTALLATION
Install required dependencies:

pip install -r requirements.txt

**Clone the repository:**

git clone https://github.com/yourusername/financial-inclusion.git

cd financial-inclusion

**Run the Streamlit application:**

streamlit run app.py

## RUN
Once the application is running, you will be presented with a form in the sidebar where you can input your demographic data, such as:

-Country

-Location type (Urban/Rural)

-Cellphone access

-Age

-Gender

-Marital status

-Education level

After filling out the form, click the "Predict" button to get the prediction on whether the individual is likely to have or use a bank account.

## DATASET
The dataset used for this project includes demographic information and financial service usage data from individuals across Kenya, Rwanda, Tanzania, and Uganda. 
It has been pre-processed for use with the machine learning model.

## MODEL DESCRIPTION
The machine learning model was trained using Scikit-learn.

The model predicts binary outcomes:

-0: The individual is likely to have or use a bank account.

-1: The individual is unlikely to have or use a bank account.

The model relies on encoded categorical variables such as country, marital status, cellphone access, etc.
Encoders for these variables are saved in separate '.sav' files and are loaded during runtime to transform user inputs into formats that the model understands.

## FILE STRUCTURE

── financial.py                  # Main application script

── requirements.txt        # List of dependencies

── openinaccount.jpg       # background image

── joblib_CountryEncoder.sav     # Pre-trained encoder for country data

── joblib_MaritalEncoder.sav     # Pre-trained encoder for marital status

── joblib_cellphone_encoder.sav  # Pre-trained encoder for cellphone access

── joblib_locationEncoder.sav    # Pre-trained encoder for location type

── joblib_genderencoder.sav      # Pre-trained encoder for gender

── joblib_educationencoder.sav   # Pre-trained encoder for education level

── joblib_clf.sav                # Pre-trained machine learning model
