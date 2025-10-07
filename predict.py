import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json


# Load all files

with open('model_lin_reg.pkl', 'rb') as file_1:
  model_lin_reg = pickle.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = pickle.load(file_2)

with open('model_encoder.pkl','rb') as file_3:
  model_encoder = pickle.load(file_3)

with open('list_num_cols.txt', 'r') as file_4:
  list_num_cols = json.load(file_4)

with open('list_cat_cols.txt', 'r') as file_5:
  list_cat_cols = json.load(file_5)

def run():
    # judul
    st.title('Prediction Player')
    # user input
    with st.form(key = 'player'):
        #input nama
        name = st.text_input('Masukan Nama Pemain',
                            placeholder = 'contoh: Cristiano Ronaldo')
        age = st.number_input('Masukan usia pemain', min_value= 0, max_value= 100,
                            value = 20)
        height = st.number_input('Masukan tinggi badan pemain', min_value= 100, max_value= 300,
                            value = 100, help = 'Tinggi dalam cm')
        weight = st.number_input('Masukan berat badan pemain', min_value= 40, max_value= 120,
                            value = 40, help = 'Berat badan dalam Kg')
        price = st.number_input('Masukan harga pemain', min_value= 0,
                            value = 500000, help = 'Harga dalam euro')
        st.write('___')
        # workrate
        attacking_wr = st.selectbox('Attacking work rate',
                                    ['Low', 'Medium', 'High'])
        defensive_wr = st.selectbox('Defensive work rate',
                                    ['Low', 'Medium', 'High'])
        
        st.write('___')
        # total columns
        pace = st.slider('Pace Total', min_value= 0, max_value= 100,
                        value= 50)
        shooting = st.slider('Shooting Total', min_value= 0, max_value= 100,
                        value= 50)
        passing = st.slider('Passing Total', min_value= 0, max_value= 100,
                        value= 50)
        dribbling = st.slider('Dribbling Total', min_value= 0, max_value= 100,
                        value= 50)
        defending = st.slider('Defending Total', min_value= 0, max_value= 100,
                        value= 50)
        physicality = st.slider('Physicality Total', min_value= 0, max_value= 100,
                        value= 50)
        
        # subnmit button
        submit = st.form_submit_button('Predict')


    if submit:    

    # Create a new data
    # Use all columns not just the results of feature selection

        data_inf = {
        'Name': name,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Price': price,
        'AttackingWorkRate': attacking_wr,
        'DefensiveWorkRate': defensive_wr,
        'PaceTotal': pace,
        'ShootingTotal': shooting,
        'PassingTotal': passing,
        'DribblingTotal': dribbling,
        'DefendingTotal': defending,
        'PhysicalityTotal':physicality
        }

        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]

        # Feature Scaling and Feature Encoding

        ## Feature Scaling
        data_inf_num_scaled = model_scaler.transform(data_inf_num)

        ## Feature Encoding
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        ## Concate
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)


        # Predict using Linear Regression
        y_pred_inf = model_lin_reg.predict(data_inf_final)
        st.write('# Prediction: ', int(y_pred_inf[0]))

if __name__ == '__main__':
    run()