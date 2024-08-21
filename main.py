# input library
import streamlit as st
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# download dataset
iris = sns.load_dataset('iris')

# define variable name
iris

#define x-axis and y-axis
x = iris.iloc[:, :-1]
y = iris['species']

# separate training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)
        
st.sidebar.title('Classifier Selection')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Neural Network'))
if classifier == 'KNN':
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'SVM':
  svm = SVC(kernel='linear')
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Decition Tree':
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = dt.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Random Forest':
  rf = RandomForestClassifier(n_estimators=100)
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Neural Network':
  nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
  nn.fit(x_train, y_train)
  y_pred = nn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
