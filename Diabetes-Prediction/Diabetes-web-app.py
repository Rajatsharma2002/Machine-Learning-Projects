import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('C:/Users/HP/ML-Projects/Diabetes-Project/Diabetes_model.sav','rb'))

def prediction(data):
    input_to_array=np.asarray(data)

    # reshaping the input
    input_reshape=input_to_array.reshape(1,-1)

    # finally predicting the value
    output=loaded_model.predict(input_reshape)

    if(output==0):
        return "Person is Non-Diabetic"
    else:
        return "Person is Diabetic"
    
def main():
    st.title("Diabetes Prediction Application")
    
  #  Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input("Enter Number of Pregnancies")
    Glucose = st.text_input("Enter Glucose level")
    BloodPressure = st.text_input("Enter Blood Pressure value")
    SkinThickness = st.text_input("Enter Skin Thickness")
    Insulin = st.text_input("Enter Insulin level")
    BMI = st.text_input("Enter BMI value")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes Pedigree Function value")
    Age = st.text_input("Enter Age")
    
    result = " "
    
    if st.button('test Result'):
        result = prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(result)
    
if __name__ == '__main__':
    main()
    