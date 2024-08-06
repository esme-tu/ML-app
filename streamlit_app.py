import streamlit as st
import pandas as pd

st.title('🧠 ML-App')
st.info('Bu uygulama makine öğrenmesi için yapılan bir uygulamadır 😊')

with st.expander('Data'):
  st.write('++Raw Data')
  df=pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
  st.write('**X**')
  X_raw= df.drop('species', axis=1)
  X_raw

  st.write('**Y**')
  Y_raw= df.species
  Y_raw
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
  data={'Ada Seçimi': island,
        'Gaga Uzunluğu (mm)': bill_length_mm,
        'Gaga Genişliği (mm)': bill_depth_mm,
        'Yüzgeç Uzunluğu (mm)': flipper_length_mm,
        'Ağırlık (g)': body_mass_g,
        'Cinsiyet': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Özellik Girişi'):
  st.write('***Özellik Girişi***')
  input_df
  st.write('**Birleştirilmiş penguen verileri**')
  input_penguins

# Data preparation
# Encode X
encode = ['Ada', 'Cinsiyet']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Veri Hazırlama'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y
# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
