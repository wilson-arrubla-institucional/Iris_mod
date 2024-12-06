#importar las librerias necesarias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

#cargar los datos en un dataset
iris = datasets.load_iris()

X = iris.data
y = iris.target

#separar los datos de entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(X,y)

#crear los objetos con las técnicas de ML
log_reg = LogisticRegression()
svc_m = SVC()

#entrenar los modelos
log_regMo = log_reg.fit(x_train, y_train)
svc_Mo = svc_m.fit(x_train, y_train )

# Guardar los modelos

with open ('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regMo, lo)
with open ('svc_Mo.pkl' ,'wb') as sv:
    pickle.dump(svc_Mo, sv)