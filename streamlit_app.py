import streamlit as st
import pandas as pd

st.title('🧠 ML-App')
st.info('Bu uygulama makine öğrenmesi için yapılan bir uygulamadır 😊')

df=pd.read_cvs('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df
