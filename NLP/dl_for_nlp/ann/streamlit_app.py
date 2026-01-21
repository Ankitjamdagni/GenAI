import streamlit as st
from pathlib import Path
import methods

pickle_file_paths = {"label_geo_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_geo_data.pkl"),
           "label_gender_encoder_path" : Path(Path.cwd(), "dl_for_nlp/Data/label_encoders/label_encoder_gender.pkl"),
           "model_scalers_path" : Path(Path.cwd(), "dl_for_nlp/Data/Churn_Modelling_scaler.pkl"),
           "model_path" : Path(Path.cwd(), "dl_for_nlp/Data/models/churn_ann_model.keras")
           }


# load encoders
encoders_and_scalers = methods.load_encoders_scalers(path=pickle_file_paths)
label_geo_encoder = encoders_and_scalers['label_geo_encoder']
label_gender_encoder = encoders_and_scalers['label_gender_encoder']
model_scalers = encoders_and_scalers['model_scalers']

# load model
chruning_model = methods.load_churning_model(model_path=pickle_file_paths["model_path"])

# streamlit_app
st.title("Customer Churning Prediction")


# defining user inputs
geography = st.selectbox('Geography', label_geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_gender_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_scalers= methods.transforms_input_data(encoders_and_scalers= encoders_and_scalers, 
                              sample_data=input_data)

# Predict churn
output = methods.predict(input_scalers=input_scalers, prediction_model=chruning_model)
prediction_proba = output[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to exit/churn.')
else:
    st.write('The customer is not likely to exit/churn.')


