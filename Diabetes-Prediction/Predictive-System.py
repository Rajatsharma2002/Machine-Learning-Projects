import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/HP/ML-Projects/Diabetes-Project/Diabetes_model.sav','rb'))

data=(4,76,62,0,0,34,0.391,25)

# coverting input to array
input_to_array=np.asarray(data)

# reshaping the input
input_reshape=input_to_array.reshape(1,-1)

# finally predicting the value
output=loaded_model.predict(input_reshape)

if(output==0):
    print("Person is Non-Diabetic")
else:
    print("Person is Diabetic")