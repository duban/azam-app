import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from st_aggrid import AgGrid
import pickle
import re


def training_data(data):
    print(data)


def upload_data_training():
    st.markdown("<h5 style='text-align: center;'>DATA TRAINING</h5>",
                unsafe_allow_html=True)
    uploaded_dataset = st.file_uploader(
        "Choose a CSV file of dataset training")
    iris = load_iris()
    # st.write(type(iris.data))
    X = iris.data
    y = iris.target
    # st.write(type(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    col1, col2 = st.columns(2)
    # col1.write(X_train)
    # col2.write(y_train)
    filename = 'gnb_test.sav'
    # if st.button('Train'):
    #     gnb = GaussianNB()
    #     gnb.fit(X_train, y_train)
    #     st.write('Train')
    #     # st.write(gnb)

    #     # save the model to disk
    # pickle.dump(gnb, open(filename, 'wb'))
    # st.write("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

    # if st.button('Test'):
    #     # load the model from disk
    #     loaded_gnb = pickle.load(open(filename, 'rb'))
    #     y_pred = loaded_gnb.predict(X_test)
    # st.write(y_pred)
    # st.write("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

    # st.write(iris.target)
    # df = pd.DataFrame(iris.data, columns = iris.feature_names)
    # st.write(df)
    # st.write(iris.data)
    # st.write(iris.feature_names)

    if uploaded_dataset is not None:
        df = pd.read_csv(uploaded_dataset)
        # print(df.columns)
        # tips = sns.load_dataset("tips")
        # st.write(tips)
        st.caption('Dataset Preview')
        # st.write(df.dropna(how='all', axis=1))
        df = df.dropna()
        # st.write(df)
        AgGrid(df, height=300)
        # st.write(df)
        X = df.loc[:, df.columns != 'Diagnosa']
        y = df.loc[:, df.columns == 'Diagnosa']
        # df_train = df[df["Diagmosa"] == chosen_range['cluster']]
        cols = X.columns
        for col in cols:
            X.loc[X[col] == 'Iya', col] = 1
            X.loc[X[col] == 'Tidak', col] = 0
            X.loc[X[col] == 'L', col] = 1
            X.loc[X[col] == 'P', col] = 0

        # st.write(X.to_numpy())
        X = X.apply(pd.to_numeric, errors='coerce')
        # st.write(X.dropna(how='all'))

        X_train, X_test, y_train, y_test = train_test_split(
            X.to_numpy(), y.to_numpy(), test_size=0.1, random_state=0)

        # st.write(len(X))
        # col1.write(X_test)
        # col2.write(y_test)

        # Every form must have a submit button.
        submitted = st.button("TRAIN")
        if submitted:
            try:
                gnb = GaussianNB()
                gnb.fit(X_train, y_train)
                
                # save the model to disk
                pickle.dump(gnb, open(filename, 'wb'))

                # loaded_gnb = pickle.load(open(filename, 'rb'))
                y_pred = gnb.predict(X_test)
                # st.write(y_pred)

                cm = confusion_matrix(y_test, y_pred)
                # print('cm:',cm)
                # st.write('YTEST', y_test)
                # st.write('YPRED', y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='micro')
                recall = recall_score(y_test, y_pred, average='micro')
                f1 = f1_score(y_test, y_pred, average='micro')
                
                data = {'accuracy':[accuracy],'precision':[precision],'recall':[recall], 'f1 score':[f1]}
                df = pd.DataFrame(data)
                
                
                hide_table_row_index = """
                    <style> 
                    tbody th {display:none}
                    .blank {display:none}
                    </style>
                    """

                st.markdown(hide_table_row_index, unsafe_allow_html=True)
                st.table(df)
                # AgGrid(df, height=70)
                # st.write('Confusion matrix for Naive Bayes : ', cm)
                # st.write('accuracy_Naive Bayes: %.3f' % accuracy)
                # st.write('precision_Naive Bayes: %.3f' % precision)
                # st.write('recall_Naive Bayes: %.3f' % recall)
                # st.write('f1-score_Naive Bayes : %.3f' % f1)
                # st.success('Upload success')
            except:
                st.error('Training failed')


def remove_duplicates(txt):
    result = []
    for word in txt.split():
        if word not in result:
            result.append(word)
    return ' '.join(result)


def testing():
    st.markdown("<h5 style='text-align: center;'>DATA TESTING</h5>",
                unsafe_allow_html=True)
    uploaded_dataset = st.file_uploader("Choose a CSV file of dataset testing")
    if uploaded_dataset is not None:
        df = pd.read_csv(uploaded_dataset)
        df2 = df.copy()
        df2 = df2.dropna()
        # print(df.columns)
        # tips = sns.load_dataset("tips")
        # st.write(tips)
        st.caption('Dataset Preview')
        # st.write(df.dropna(how='all', axis=1))
        # df = df.dropna()
        # df = df.loc[:, df.columns != 'Nama Pasien']
        # df2=df.loc[(df.columns != 'Nomor RM') & (df.columns != 'Nama Pasien')]
        df = df.loc[:,(df.columns != 'Nomor Rawat') & (df.columns != 'Nama Pasien')]
        X = df.dropna()
        AgGrid(df2, height=300)
        # st.write(df2)
        # st.write(df)
        # df_train = df[df["Diagmosa"] == chosen_range['cluster']]
        cols = df.columns
        for col in cols:
            X.loc[X[col] == 'Iya', col] = 1
            X.loc[X[col] == 'Tidak', col] = 0
            X.loc[X[col] == 'L', col] = 1
            X.loc[X[col] == 'P', col] = 0

            # st.write(X.to_numpy())
        X_test = X.apply(pd.to_numeric, errors='coerce')
        # st.write(X)
        submitted = st.button("PREDICT")
        if submitted:
            try:
                # gnb/ = GaussianNB()
                # gnb.fit(X_train, y_train)
                # save the model to disk
                # pickle.dump(gnb, open(filename, 'wb'))

                loaded_gnb = pickle.load(open('gnb_test.sav', 'rb'))
                y_pred = loaded_gnb.predict(X_test.to_numpy())
                df_diagnosa = pd.DataFrame(y_pred, columns=['Prediksi Diagnosa'])
                combined_df = pd.concat([df2[['Nama Pasien','Usia','Jenis Kelamin']],df_diagnosa], axis=1, join="inner")
                
                df_full = pd.concat([df2,df_diagnosa], axis=1, join="inner")
                df_full.to_csv('./data/results.csv', encoding='utf-8', index=False)
        
                AgGrid(df_full, height=400)
                # st.write(type(y_pred))
                    # conn = db_config()
                    # df.to_sql('dataset_mcn', con=conn,
                    #           if_exists='append', index=False)
                # st.success('Upload success')
            except:
                st.error('Upload failed')
# Using "with" notation

def list_pasien():
    st.markdown("<h5 style='text-align: center;'>DAFTAR PASIEN</h5>",
                unsafe_allow_html=True)
    try:
        df = pd.read_csv('./data/results.csv')
        # st.write(df)
        AgGrid(df)
    except:
        st.info('Results not found!')

if __name__ == "__main__":
    with st.sidebar:
        selected_menu = st.radio(
            "Choose Menu",
            ("Data Training", "Data Testing","Daftar Pasien"))

    if selected_menu == 'Data Training':
        upload_data_training()
    if selected_menu == 'Data Testing':
        testing()
    if selected_menu == 'Daftar Pasien':
        list_pasien()
    
