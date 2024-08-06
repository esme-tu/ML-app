import streamlit as st
import pandas as pd

st.title('🧠 ML-App')
st.info('Bu uygulama makine öğrenmesi için yapılan bir uygulamadır 😊')

with st.expander('Data'):
  st.write('++Raw Data')
  df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
  st.write('**X**')
  X= df.drop('species', axis=1)
  X

  st.write('**Y**')
  Y= df.species
  Y
#"bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"
with st.expander('Data Görselleştirme'):
  st.scatter_chart(data=df,x="bill_length_mm", y="body_mass_g", color='species')

