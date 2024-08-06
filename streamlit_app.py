import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🧠 ML-App')
st.info('Bu uygulama makine öğrenmesi için yapılan bir uygulamadır 😊')

with st.expander('Data'):
    st.write('++Raw Data')
    df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.write(df)
    st.write('**X**')
    X_raw = df.drop('species', axis=1)
    st.write(X_raw)

    st.write('**Y**')
    y_raw = df.species
    st.write(y_raw)

with st.expander('Data Görselleştirme'):
    st.scatter_chart(data=df, x="bill_length_mm", y="body_mass_g", color='species')

# Özellik girişi
with st.sidebar:
    st.header('Özellik Girişi')
    island = st.selectbox('Ada Seçimi', ('Biscoe', 'Dream', 'Torgersen'))
    gender = st.selectbox('Cinsiyet', ('female', 'male'))
    bill_length_mm = st.slider('Gaga Uzunluğu (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Gaga Genişliği (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Yüzgeç Uzunluğu (mm)', 170.0, 231.0, 201.0)
    body_mass_g = st.slider('Ağırlık (g)', 2700.0, 6300.0, 4207.0)

    # Girilen özelliğe göre veri oluştur
    data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Özellik Girişi'):
    st.write('***Özellik Girişi***')
    st.write(input_df)
    st.write('**Birleştirilmiş penguen verileri**')
    st.write(input_penguins)

# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, columns=encode)

X = df_penguins.iloc[1:, :]
input_row = df_penguins.iloc[:1, :]

# Encode y
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.apply(lambda val: target_mapper[val])

with st.expander('Veri Hazırlama'):
    st.write('**Encoded X (input penguin)**')
    st.write(input_row)
    st.write('**Encoded y**')
    st.write(y)

# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba.style.format({'Adelie': '{:.2%}', 'Chinstrap': '{:.2%}', 'Gentoo': '{:.2%}'}))

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f'Tahmin edilen tür: {penguins_species[prediction][0]}')
