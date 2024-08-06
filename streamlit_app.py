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

#data seçme
with st.sidebar:
  st.header('Özellik Girişi')
  island = st.selectbox('Ada Seçimi',('Biscoe','Dream','Torgersen'))
  gender= st.selectbox('Cinsiyet',('female','male'))
  bill_length_mm= st.slider('Gaga Uzunluğu (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm= st.slider('Gaga Genişliği (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm= st.slider('Yüzgeç Uzunluğu (mm)', 170.0, 231.0, 201.0)
  body_mass_g= st.slider('Ağırlık (g)', 2700.0, 6300.0, 4207.0)
  
  # girilen özelliğe göre veri oluştur
  data={'Ada Seçimi', island,
        'Gaga Uzunluğu (mm)', bill_length_mm,
        'Gaga Genişliği (mm)', bill_depth_mm,
        'Yüzgeç Uzunluğu (mm)', flipper_length_mm,
        'Ağırlık (g)',body_mass_g,
        'Cinsiyet',gender}
  input_df= pd.DataFrame(data, index[0])

input_df
