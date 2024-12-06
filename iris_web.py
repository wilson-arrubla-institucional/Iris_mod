# Importar la librerias
import streamlit as st
import pickle
import pandas as pd

# Extraer los archivos pkl
with open ('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)
with open ('svc_Mo.pkl' ,'rb') as sv:
    log_svc_Mo= pickle.load(sv)

#funcion para clasificar las plantas

def clasify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'
    


def main():
    #Titulo
    st.title('Modelamiento de Iris by Wilson Arrubla')
    #Titulo de Sidebar
    st.sidebar.header('User Input Parameters')

 # Entradas del usuario en el Sidebar
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
        data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()
    # Seleccionar el modelo 
    option = ['Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('Which model you like to use?',option)

    st.subheader('User Input user input features')
    st.subheader(model)
    st.write(df)

    #Crar un Boton
    if st.button('Run'):
        if model == 'Logistic Regression':
            st.success(clasify(log_reg.predict(df)))
        else:
            st.success(clasify(log_svc_Mo.predict(df)))
if __name__ == '__main__':
    main()