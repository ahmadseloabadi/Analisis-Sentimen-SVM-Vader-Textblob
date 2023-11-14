#import library
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
import random as rd
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score


# Set page layout and title
st.set_page_config(page_title="textblob", page_icon="img/icon.png")

#import dataset
dataset_manual = pd.read_csv('data/dataset/sentimen_manual.csv')

#import dataset
dataset_textblob = pd.read_csv('data/dataset/sentimen_textblob.csv')

# text preprosessing
def cleansing(kalimat_baru): 
    kalimat_baru = re.sub(r'@[A-Za-a0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r'#[A-Za-z0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"http\S+",' ',kalimat_baru)
    kalimat_baru = re.sub(r'[0-9]+',' ',kalimat_baru)
    kalimat_baru = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", kalimat_baru)
    kalimat_baru = re.sub(r"\b[a-zA-Z]\b", " ", kalimat_baru)
    kalimat_baru = kalimat_baru.strip(' ')
    # menghilangkan emoji
    def clearEmoji(ulasan):
        return ulasan.encode('ascii', 'ignore').decode('ascii')
    kalimat_baru =clearEmoji(kalimat_baru)
    def replaceTOM(ulasan):
        pola = re.compile(r'(.)\1{2,}', re.DOTALL)
        return pola.sub(r'\1', ulasan)
    kalimat_baru=replaceTOM(kalimat_baru)
    return kalimat_baru
def casefolding(kalimat_baru):
    kalimat_baru = kalimat_baru.lower()
    return kalimat_baru
def tokenizing(kalimat_baru):
    kalimat_baru = word_tokenize(kalimat_baru)
    return kalimat_baru
def slangword (kalimat_baru):
    kamusSlang = eval(open("data/kamus/slangwords.txt").read())
    pattern = re.compile(r'\b( ' + '|'.join (kamusSlang.keys())+r')\b')
    content = []
    for kata in kalimat_baru:
        filter_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata.lower())
        if filter_slang.startswith('tidak_'):
          kata_depan = 'tidak_'
          kata_belakang = kata[6:]
          kata_belakang_slang = pattern.sub(lambda x: kamusSlang[x.group()], kata_belakang.lower())
          kata_hasil = kata_depan + kata_belakang_slang
          content.append(kata_hasil)
        else:
          content.append(filter_slang)
    kalimat_baru = content
    return kalimat_baru
def handle_negation(kalimat_baru):
    negation_words = ["tidak", "bukan", "tak", "tiada", "jangan", "gak",'ga']
    new_words = []
    prev_word_is_negation = False
    for word in kalimat_baru:
        if word in negation_words:
            new_words.append("tidak_")
            prev_word_is_negation = True
        elif prev_word_is_negation:
            new_words[-1] += word
            prev_word_is_negation = False
        else:
            new_words.append(word)
    return new_words
def stopword (kalimat_baru):
    daftar_stopword = stopwords.words('indonesian')
    daftar_stopword.extend(["yg", "dg", "rt", "dgn", "ny", "d",'gb','ahk','g','anjing','ga','gua','nder']) 
    # Membaca file teks stopword menggunakan pandas
    txt_stopword = pd.read_csv("data/kamus/stopwords.txt", names=["stopwords"], header=None)

    # Menggabungkan daftar stopword dari NLTK dengan daftar stopword dari file teks
    daftar_stopword.extend(txt_stopword['stopwords'].tolist())

    # Mengubah daftar stopword menjadi set untuk pencarian yang lebih efisien
    daftar_stopword = set(daftar_stopword)

    def stopwordText(words):
        cleaned_words = []
        for word in words:
            # Memisahkan kata dengan tambahan "tidak_"
            if word.startswith("tidak_"):
                cleaned_words.append(word[:5])
                cleaned_words.append(word[6:])
            elif word not in daftar_stopword:
                cleaned_words.append(word)
        return cleaned_words
    kalimat_baru = stopwordText(kalimat_baru)
    return kalimat_baru 
def stemming(kalimat_baru):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Lakukan stemming pada setiap kata
    stemmed_words = [stemmer.stem(word) for word in kalimat_baru]
    return stemmed_words


# mengambil kolom Stopword Removal dan Sentiment pada dataset
X_manual = dataset_manual['Stopword Removal']
Y_manual = dataset_manual['Sentimen']

vectorizer = TfidfVectorizer()

X_textblob = dataset_textblob['Stopword Removal']
Y_textblob = dataset_textblob['Sentimen']


def train_sentiment_svm(data, labels):
    # Membagi data menjadi data pelatihan dan data uji
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Membuat vektor fitur dari data teks dengan TF-IDF

    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Melatih model SVM
    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train_tfidf, y_train)

    return svm_classifier, vectorizer

def predict_sentiment(model, vectorizer, new_sentences):
    # Menerapkan vektorisasi TF-IDF ke kalimat baru
    new_sentences_tfidf = vectorizer.transform([new_sentences])

    # Melakukan prediksi sentimen
    predictions = model.predict(new_sentences_tfidf)

    return predictions



#side bar
with st.sidebar :
    selected = option_menu('sentimen analisis',['Home','Pengolahan data','Uji','Report'])

if(selected == 'Home') :
    st.title('ANALISIS SENTIMEN ULASAN APLIKASI SHOPEE DENGAN METODE SVM DAN TEXTBLOB ')
    st.write('Shopee adalah platform belanja online terdepan di Asia Tenggara dan Taiwan. Diluncurkan tahun 2015, Shopee merupakan sebuah platform yang disesuaikan untuk tiap wilayah dan menyediakan pengalaman berbelanja online yang mudah, aman, dan cepat bagi pelanggan melalui dukungan pembayaran dan logistik yang kuat.')
    # import gambar pada halaman home
    image = Image.open('img/Shopee.png')
    st.image(image)
    st.title('dataset ulasan SHOPEE')
    st.write('dataset diambil dari situs kaggle')
    dataset_awal = pd.read_csv('data/dataset/shopee_ulasan_label7.csv')
    st.dataframe(dataset_awal)

elif(selected == 'Pengolahan data') :
    tab1,tab2=st.tabs(['pelabelan','Text preprosesing'])
    # di tab pelabelan ini kita buat untuk nampilin dataset sentimen manual dan dataset sentimen textblob
    with tab1 :
        # disini kita bikin selectbox/dropdown buat milih dataset yang mau di tampilin
        menu_dataset = st.selectbox('pelabelan dataset',('Manual', 'textblob'))
        # ini bagian untuk nampilin dataset sentimen manual
        if (menu_dataset =='Manual'): 
            st.write('dataset ulasan SHOPEE dengan pelabelan secara menual')
            # import dataset sentimen manual
            dataset = pd.read_csv('data/dataset/sentimen_manual.csv')
            # membuat fungsi untuk sentimen pada dataset
            def sentimen_manual(dataset, sentiment):
                return dataset[dataset['Sentimen'].isin(sentiment)]
            # inisialisasi nilai pada kolom sentimen 
            sentiment_map = {'positif': 'sentimen positif', 'netral': 'sentimen netral','negatif':'sentimen negatif'}
            sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
            sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
            filtered_data = sentimen_manual(dataset, sentiment)
            st.dataframe(filtered_data,use_container_width=True)
            # Hitung jumlah kelas dataset
            st.write("Jumlah kelas:  ")
            kelas_sentimen = dataset_manual['Sentimen'].value_counts()
            # st.write(kelas_sentimen)
            datpos, datneg, datnet = st.columns(3)
            with datpos:
                st.markdown("Positif")
                st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
            with datnet:
                st.markdown("Netral")
                st.markdown(f"<h1 style='text-align: center; color: orange;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
            with datneg:
                st.markdown("Negatif")
                st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
            #membuat diagram
            labels = ['negatif' , 'neutral', 'positif']
            fig1,ax1=plt.subplots()
            ax1.pie(dataset_manual.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
            ax1.axis('equal')
            st.pyplot(fig1)
            # ini buat nampilin dataset sentimen textblob /automatis 
        elif (menu_dataset =='textblob') :
            st.write('dataset ulasan SHOPEE dengan pelabelan menggunakan textblob based')
            dataset = pd.read_csv('data/dataset/sentimen_textblob.csv')
            def sentimen_textblob(dataset, sentiment):
                return dataset[dataset['Sentimen'].isin(sentiment)]
            sentiment_map = {'Positive': 'sentimen positif', 'Netral': 'sentimen netral','Negative':'sentimen negatif'}
            sentiment = st.multiselect('Pilih kelas sentimen', list(sentiment_map.values()),default=list(sentiment_map.values()))
            sentiment = [key for key, value in sentiment_map.items() if value in sentiment]
            filtered_data = sentimen_textblob(dataset, sentiment)
            st.dataframe(filtered_data,use_container_width=True)
            # Hitung jumlah kelas dataset
            st.write("Jumlah kelas:  ")
            kelas_sentimen = dataset_textblob['Sentimen'].value_counts()
            # st.write(kelas_sentimen)
            datpos, datneg, datnet = st.columns(3)
            with datpos:
                st.markdown("Positif")
                st.markdown(f"<h1 style='text-align: center; color: green;'>{kelas_sentimen[0]}</h1>", unsafe_allow_html=True)
            with datnet:
                st.markdown("Netral")
                st.markdown(f"<h1 style='text-align: center; color: orange;'>{kelas_sentimen[2]}</h1>", unsafe_allow_html=True)
            with datneg:
                st.markdown("Negatif")
                st.markdown(f"<h1 style='text-align: center; color: blue;'>{kelas_sentimen[1]}</h1>", unsafe_allow_html=True)
            #membuat diagram
            labels = ['negatif' , 'neutral', 'positif']
            fig1,ax1=plt.subplots()
            ax1.pie(dataset_manual.groupby('Sentimen')['Sentimen'].count(), autopct=" %.1f%% " ,labels=labels)
            ax1.axis('equal')
            st.pyplot(fig1)
    with tab2 :
        # tab2 ini akan menampilkan hasil preprosessing yang udh kita lakuin di kodingan mentah
        st.title('Text preprosesing')
        st.header('casefolding')#----------------
        st.text('mengubahan seluruh huruf menjadi kecil (lowercase) yang ada pada dokumen.')
        # menginport data hasil casefolding 
        casefolding = pd.read_csv('data/prepro/casefolding.csv')
        st.write(casefolding)
        st.header('cleansing')#----------------
        st.text('membersihkan data dari angka ,tanda baca,dll.')
        cleansing = pd.read_csv('data/prepro/cleansing.csv')
        st.write(cleansing)
        st.header('tokenizing')#----------------
        st.text('menguraikan kalimat menjadi token-token atau kata-kata.')
        tokenizing = pd.read_csv('data/prepro/tokenizing.csv')
        st.write(tokenizing)
        st.header('word normalization')#----------------
        st.text('mengubah penggunaan kata tidak baku menjadi baku')
        word_normalization = pd.read_csv('data/prepro/word normalization.csv')
        st.write(word_normalization)
        st.header('stopword')#----------------
        st.text('menyeleksi kata yang tidak penting dan menghapus kata tersebut.')
        stopword = pd.read_csv('data/prepro/stopword.csv')
        st.write(stopword)
        st.header('stemming')#----------------
        st.text(' merubahan kata yang berimbuhan menjadi kata dasar. ')
        stemming = pd.read_csv('data/prepro/stemming.csv')
        st.write(stemming)
        st.title('Pembobotan TF-IDF')
        st.text('pembobotan pada penelitan ini menggunakan tf-idf')
        tfidf = pd.read_csv('data/prepro/hasil TF IDF.csv')
        st.dataframe(tfidf,use_container_width=True) 
elif(selected == 'Uji') :
    opsi1,opsi2=st.tabs(['opsi tampilan 1','opsi tampilan 2'])
    with opsi1 :
        opsi_metode = st.selectbox('METODE',('SVM', 'SVM-textblob'))
        kalimat_baru = st.text_input('masukan kalimat',value="tingkatkan terus kalau bisa ada kerjasama dengan paylater")
        
        kcleansing = cleansing(kalimat_baru)
        kcasefolding = casefolding(kcleansing)
        ktokenizing = tokenizing(kcasefolding)
        kstemming = stemming(ktokenizing)
        knegasi= handle_negation(kstemming)
        kslangword = slangword(knegasi)
        kstopword = stopword(kslangword)
        kdatastr = str(kstopword)
        # ktfidf =vectorizer.transform([kdatastr])

        if (opsi_metode == 'SVM') :
            if st.button('predik') :
                st.write('Hasil pengujian dengan metode',opsi_metode)
                # Making the SVM Classifer
                # Melatih model
                model, vectorizer = train_sentiment_svm(X_manual, Y_manual)
                predictions = predict_sentiment(model, vectorizer, kdatastr)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stemming :',str(kstemming))
                st.write('hasil negasi :',str(knegasi))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stopword :',str(kstopword))
                st.write(f"hasil prediksi menggunakan metode {opsi_metode} adalah {predictions}")
            else:
                st.write('Hasil') 

        elif (opsi_metode == 'SVM-textblob') :
            if st.button('predik') :
                st.write('Hasil pengujian dengan metode',opsi_metode)
                # Making the SVM Classifer
                model, vectorizer = train_sentiment_svm(X_textblob, Y_textblob)
                predictions = predict_sentiment(model, vectorizer, kdatastr)
                st.write('hasil cleansing :',str(kcleansing))
                st.write('hasil casefolding :',str(kcasefolding))
                st.write('hasil tokenizing :',str(ktokenizing))
                st.write('hasil stemming :',str(kstemming))
                st.write('hasil negasi :',str(knegasi))
                st.write('hasil word normalization :',str(kslangword))
                st.write('hasil stopword :',str(kstopword))
                st.write(f"hasil prediksi menggunakan metode {opsi_metode} adalah {predictions}")
            else:
                st.write('Hasil') 
    with opsi2 :
        kalimat_baru = st.text_input('masukan kalimat',value="tingkatkan terus kalau bisa ada kerjasama dengan paylater.")
        
        kcleansing = cleansing(kalimat_baru)
        kcasefolding = casefolding(kcleansing)
        ktokenizing = tokenizing(kcasefolding)
        kstemming = stemming(ktokenizing)
        knegasi= handle_negation(kstemming)
        kslangword = slangword(knegasi)
        kstopword = stopword(kslangword)
        kdatastr = str(kstopword)
        
        if st.button('prediksi') :
            model_manual, vectorizer_manual = train_sentiment_svm(X_manual, Y_manual)
            predictions_manual = predict_sentiment(model_manual, vectorizer_manual, kdatastr)
            model_textblob, vectorizer_textblob = train_sentiment_svm(X_textblob, Y_textblob)
            predictions_textblob = predict_sentiment(model_textblob, vectorizer_textblob, kdatastr)
            st.write('hasil cleansing :',str(kcleansing))
            st.write('hasil casefolding :',str(kcasefolding))
            st.write('hasil tokenizing :',str(ktokenizing))
            st.write('hasil stemming :',str(kstemming))
            st.write('hasil negasi :',str(knegasi))
            st.write('hasil word normalization :',str(kslangword))
            st.write('hasil stopword :',str(kstopword))
            st.write(f"hasil prediksi menggunakan metode SVM adalah pelabelan manual {predictions_manual}")
            st.write(f"hasil prediksi menggunakan metode SVM adalah pelabelan textblob {predictions_textblob}")
        else:
            st.write('Hasil') 

elif(selected == 'Report') :

    tab1,tab2 =st.tabs(['klasifikasi report','confusion matrix'])

    with tab1 :
        st.header('perbandingan nilai accuracy,precision,dan recall pada dataset manual dan dataset textblob')
        hasil_evaluasi = pd.read_csv('data/eval/hasil_evaluasi.csv')
        st.write(hasil_evaluasi)

    with tab2 :
        # plot confusion matrix svm
        st.title('evaluasi model')
        st.header('Confusion matriks metode manual-svm')
        cm_manual = Image.open('data/eval/confusion_elsa.png')
        st.image(cm_manual)

        # plot confusion matrix smote svm
        st.header('Confusion matriks metode textblob-svm')
        cm_textblob = Image.open('data/eval/confusion_elsa.png')
        st.image(cm_textblob)




