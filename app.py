import joblib
import streamlit as st

model = joblib.load(r"Model\model_logistic_regression.pkl")
tfidf = joblib.load(r"Model\tfidf_vectorizer.pkl")

st.title("Aplikasi Klasifikasi Komentar Publik")
st.write("Selamat datang di aplikasi klasifikasi komentar publik menggunakan model Logistic Regression! Aplikasi ini dibuat menggunakan Teknologi NLP dengan memanfaatkan model machine learning logistic regression ")
input = st.text_area("Masukkan komentar Anda!!:")
if st.button("Submit"):
    if input.strip() == "":
        st.warning("Komentar tidak boleh kosong!")
    else:
        vector = tfidf.transform([input])
        prediksi = model.predict(vector)[0]
        label_map = {0: "Negatif", 1: "Positif"}
        st.subheader(f"Komentar Anda diklasifikasikan sebagai:")
        st.write("**komentar :**", label_map.get(prediksi, prediksi))

   
    

