import streamlit as st
import eda, predict

with st.sidebar:
    st.title('Page Navigation')
    # input
    page = st.radio('Page', ('EDA', 'Model Demo'))

    st.write('# About')
    st.write(''' Page ini adalah informasi data dan demo dari model
             prediksi player''')
    
if page == "EDA":
    eda.run()
else:
    predict.run()