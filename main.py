import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

df = pd.read_csv('heart.csv')
x = df.drop(columns='target', axis=1)
y = df.target
print(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=38)

model = LogisticRegression(C=0.9, class_weight='balanced', max_iter=100, solver='lbfgs')
model.fit(x, y)
print(model.score(X_test, y_test))  # accuracy_score = 0.858

def main():
    st.title('CARDIOVASCULAR DISEASE PREDICTOR')

    age = st.number_input('AGE:')
    gender = st.number_input('GENDER:')
    cp = st.number_input('chest pain type:')
    restbp = st.number_input('rest blood pressure:')
    chol = st.number_input('Cholestrol:')
    fbs = st.number_input('fasting blood sugar:')
    restecg = st.number_input('REST ECG RESULT:')
    thalach = st.number_input('maximum heart rate achieved:')
    exang = st.number_input('exercise induced angina:')
    oldpeak = st.number_input('OLD PEAK:')
    slope = st.number_input('SLOPE:')
    ca = st.number_input('Number of major vessels colored by florosopy:')
    thal = st.number_input('THAL(0/1/2):')

    if st.button('submit'):
       a = model.predict([[age, gender, cp, restbp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
       st.success(a)

if __name__ == '__main__':
 main()



