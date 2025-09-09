import json
import os
import gdown
import streamlit as st
from google.oauth2.service_account import Credentials
import gspread
import datetime
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests, zipfile, io
from tensorflow.keras.saving import load_model
import sklearn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import AUC
from keras.models import load_model
from xgboost import XGBClassifier

import tensorflow as tf

# Streamlit 版本
# st.write("**Streamlit version:**", st.__version__)

#超重要，model的threshold (目前沒用到)
AKD_optimal_threshold = 0.28
AKI_optimal_threshold = 0.44

# Load the AKD model
@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/ChanWeiKai0118/Research/main/AKD-LSTM.zip"
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(".")
    model = load_model("AKD-LSTM.keras", compile=False)
    return model

# Load the AKD scaler
@st.cache_resource
def get_scaler():
    scaler_url = "https://raw.githubusercontent.com/ChanWeiKai0118/Research/main/Scaler-AKD.pkl"
    scaler_response = requests.get(scaler_url)
    with open("Scaler-AKD.pkl", "wb") as scaler_file:
        scaler_file.write(scaler_response.content)
    return joblib.load("Scaler-AKD.pkl")

# Load the AKD imputation
@st.cache_resource
def get_imputer():
    url = "https://raw.githubusercontent.com/ChanWeiKai0118/Research/main/MICE-AKD.pkl"
    response = requests.get(url)
    with open("MICE-AKD.pkl", "wb") as imputer_file:
        imputer_file.write(response.content)
    return joblib.load("MICE-AKD.pkl")

# Load the AKI model
@st.cache_resource
def get_aki_model():
    FILE_ID = "1Y7wo3oueuzw9X-ChrbpHScLnfHIkFVJp"
    zip_path = "AKI_model.zip"
    extract_folder = "AKI_model"

    # 下載 ZIP
    if not os.path.exists(zip_path):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, zip_path, quiet=False)

    # 解壓縮
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)

    # 找到 .keras 檔案
    keras_files = [f for f in os.listdir(extract_folder) if f.endswith(".keras")]
    if len(keras_files) == 0:
        raise FileNotFoundError("Cannot find .keras file in the ZIP.")

    model_path = os.path.join(extract_folder, keras_files[0])
    model = load_model(model_path, compile=False)
    return model

# Load the AKI scaler
@st.cache_resource
def get_aki_scaler():
    aki_scaler_url = "https://raw.githubusercontent.com/ChanWeiKai0118/Research/main/Scaler-AKI.pkl"
    aki_scaler_response = requests.get(aki_scaler_url)
    with open("Scaker-AKI.pkl", "wb") as aki_scaler_file:
        aki_scaler_file.write(aki_scaler_response.content)
    return joblib.load("Scaker-AKI.pkl")

# Load the AKI imputation
@st.cache_resource
def get_aki_imputer():
    aki_url = "https://raw.githubusercontent.com/ChanWeiKai0118/Research/main/MICE-AKI.pkl"
    aki_response = requests.get(aki_url)
    with open("MICE-AKI.pkl", "wb") as aki_file:
        aki_file.write(aki_response.content)
    return joblib.load("MICE-AKI.pkl")

def post_sequential_padding( # (for return_sequences True)
        data, groupby_col, selected_features, outcome, maxlen
    ):
    grouped = data.groupby(groupby_col)
    sequences = []
    labels = []
    for name, group in grouped:
        sequences.append(group[selected_features].values)
        labels.append(group[[outcome]].values)

    X = pad_sequences(
        sequences,
        maxlen=maxlen,
        dtype='float32',
        padding='post',
        truncating='post',
        value=-1
    )

    y = pad_sequences(
        labels,
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=-1
    )

    return X, y
        
def preprocessing(
        data, scaler, imputer, cols_for_preprocessing,
        selected_features, groupby_col, outcome, maxlen
    ):
    # passing arguments
    test = data
    scaler_ = scaler
    imputer_ = imputer

    # feature selection
    test_selected = test[cols_for_preprocessing]

    # imputation
    test_imputed = test_selected.copy()
    test_imputed[selected_features] = imputer_.transform(test_selected[selected_features])

    # scaling
    test_scaled = test_imputed.copy()
    test_scaled[selected_features] = scaler_.transform(test_imputed[selected_features])

    # sequential padding
    X_test, y_test = post_sequential_padding(
        data=test_scaled,
        groupby_col=groupby_col,
        selected_features=selected_features,
        outcome=outcome,
        maxlen=maxlen
    )

    return X_test, y_test


def get_gsheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["google_service_account"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
    client = gspread.authorize(creds)
    return client

def save_to_gsheet(data, sheet_name):
    client = get_gsheet_client()
    
    if sheet_name == "chemo_data":
        sheet = client.open("web data").worksheet("chemo_data")
        row = ["" for _ in range(67)]  
        row[1], row[3], row[2], row[4], row[5] = data[0], data[1], data[2], data[3], data[4]
    
        if data[7] != 0:
            row[6], row[7] = data[5], 0
        else:
            row[6], row[7] = 0, data[5]
    
        row[10], row[13] = data[7], data[8]
        
        # 抓之前的資料
        all_rows = sheet.get_all_values() 
        last_row = len(all_rows) + 1
    
        row[0] = f'=IF(ROW()=2, 1, IF(COUNTIF(B$2:B{last_row-1}, B{last_row}) = 0, MAX(A$2:A{last_row-1}) + 1, IF(OR(H{last_row}<INDEX(H$2:H{last_row-1}, MAX(IF($B$2:B{last_row-1}=B{last_row}, ROW($B$2:B{last_row-1})-1, 0))),G{last_row}<INDEX(G$2:G{last_row-1}, MAX(IF($B$2:B{last_row-1}=B{last_row}, ROW($B$2:B{last_row-1})-1, 0))),F{last_row} - INDEX(F$2:F{last_row-1}, MAX(IF($B$2:B{last_row-1} = B{last_row}, ROW($B$2:B{last_row-1}) - 1, 0))) > 42), MAX(A$2:A{last_row-1}) + 1, INDEX(A$2:A{last_row-1}, MAX(IF(B$2:B{last_row-1}=B{last_row}, ROW($B$2:B{last_row-1})-1, 0))))))'
    
        row[8] = f'=IF(COUNTIF(A$2:A{last_row}, A{last_row}) = 1, 0, (F{last_row} - INDEX(F$2:F{last_row}, MATCH(A{last_row}, A$2:A{last_row}, 0)))/7)'
        row[9] = data[6]
        
        row[11] = f'=SUMIF(A$2:A{last_row}, A{last_row}, K$2:K{last_row})'
        row[12] = f'=IF(OR(G{last_row}=0, L{last_row}=0), 0, L{last_row} / G{last_row})'
        row[14] = f'=SUMIF(A$2:A{last_row}, A{last_row}, N$2:N{last_row})'
        row[15] = f'=IF(OR(H{last_row}=0, O{last_row}=0), 0, O{last_row} / H{last_row})'
        row[16] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(R{last_row} <> "", R{last_row},IFNA(INDEX(lab_data!H:H,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!H:H <> "")))),0)), "")),INDEX(Q$2:Q{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[17] = f'=IFNA(INDEX(lab_data!H:H, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!H:H <> "")))) * (lab_data!H:H <> ""), 0)), "")'
        row[18] = f'=IF(OR(Q{last_row}="",T{last_row}=""),"",IF(T{last_row}=0,0,T{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!H:H<>""))))*(lab_data!H:H<>""),0))-XLOOKUP(Q{last_row},FILTER(lab_data!H:H,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[19] = f'=IF(R{last_row}="", "", R{last_row} - Q{last_row})'
        row[20] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(V{last_row} <> "", V{last_row},IFNA(INDEX(lab_data!J:J,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!J:J <> "")))),0)), "")),INDEX(U$2:U{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[21] = f'=IFNA(INDEX(lab_data!J:J, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!J:J <> "")))) * (lab_data!J:J <> ""), 0)), "")'
        row[22] = f'=IF(OR(U{last_row}="",X{last_row}=""),"",IF(X{last_row}=0,0,X{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!J:J<>""))))*(lab_data!J:J<>""),0))-XLOOKUP(U{last_row},FILTER(lab_data!J:J,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[23] = f'=IF(V{last_row}="", "", V{last_row} - U{last_row})'
        row[24] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(Z{last_row} <> "", Z{last_row},IFNA(INDEX(lab_data!K:K,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!K:K <> "")))),0)), "")),INDEX(Y$2:Y{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[25] = f'=IFNA(INDEX(lab_data!K:K, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!K:K <> "")))) * (lab_data!K:K <> ""), 0)), "")'
        row[26] = f'=IF(OR(Y{last_row}="",AB{last_row}=""),"",IF(AB{last_row}=0,0,AB{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!K:K<>""))))*(lab_data!K:K<>""),0))-XLOOKUP(Y{last_row},FILTER(lab_data!K:K,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[27] = f'=IF(Z{last_row}="", "", Z{last_row} - Y{last_row})'
        row[28] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(AD{last_row} <> "", AD{last_row},IFNA(INDEX(lab_data!G:G,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!G:G <> "")))),0)), "")),INDEX(AC$2:AC{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[29] = f'=IFNA(INDEX(lab_data!G:G, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!G:G <> "")))) * (lab_data!G:G <> ""), 0)), "")'
        row[30] = f'=IF(OR(AC{last_row}="",AF{last_row}=""),"",IF(AF{last_row}=0,0,AF{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!G:G<>""))))*(lab_data!G:G<>""),0))-XLOOKUP(AC{last_row},FILTER(lab_data!G:G,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[31] = f'=IF(AD{last_row}="", "", AD{last_row} - AC{last_row})'
        row[32] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(AH{last_row} <> "", AH{last_row},IFNA(INDEX(lab_data!I:I,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!I:I <> "")))),0)), "")),INDEX(AG$2:AG{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[33] = f'=IFNA(INDEX(lab_data!I:I, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!I:I <> "")))) * (lab_data!I:I <> ""), 0)), "")'
        row[34] = f'=IF(OR(AG{last_row}="",AJ{last_row}=""),"",IF(AJ{last_row}=0,0,AJ{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!I:I<>""))))*(lab_data!I:I<>""),0))-XLOOKUP(AG{last_row},FILTER(lab_data!I:I,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[35] = f'=IF(AH{last_row}="", "", AH{last_row} - AG{last_row})'
        row[36] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(AL{last_row} <> "", AL{last_row},IFNA(INDEX(lab_data!L:L,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!L:L <> "")))),0)), "")),INDEX(AK$2:AK{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[37] = f'=IFNA(INDEX(lab_data!L:L, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!L:L <> "")))) * (lab_data!L:L <> ""), 0)), "")'
        row[38] = f'=IF(OR(AK{last_row}="",AN{last_row}=""),"",IF(AN{last_row}=0,0,AN{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!L:L<>""))))*(lab_data!L:L<>""),0))-XLOOKUP(AK{last_row},FILTER(lab_data!L:L,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[39] = f'=IF(AL{last_row}="", "", AL{last_row} - AK{last_row})'
        row[40] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(AP{last_row} <> "", AP{last_row},IFNA(INDEX(lab_data!M:M,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!M:M <> "")))),0)), "")),INDEX(AO$2:AO{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[41] = f'=IFNA(INDEX(lab_data!M:M, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!M:M <> "")))) * (lab_data!M:M <> ""), 0)), "")'
        row[42] = f'=IF(OR(AO{last_row}="",AR{last_row}=""),"",IF(AR{last_row}=0,0,AR{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!M:M<>""))))*(lab_data!M:M<>""),0))-XLOOKUP(AO{last_row},FILTER(lab_data!M:M,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[43] = f'=IF(AP{last_row}="", "", AP{last_row} - AO{last_row})'
        row[44] = f'=IF(MATCH(A{last_row}, A$2:A{last_row}, 0) = ROW()-1,IF(AT{last_row} <> "", AT{last_row},IFNA(INDEX(lab_data!N:N,MATCH(1,(lab_data!A:A = B{last_row}) *(lab_data!E:E = MAX(FILTER(lab_data!E:E,(lab_data!A:A = B{last_row}) *(lab_data!E:E <= F{last_row}) *(lab_data!E:E >= F{last_row} - 30) *(lab_data!N:N <> "")))),0)), "")),INDEX(AS$2:AS{last_row-1}, MATCH(A{last_row}, A$2:A{last_row-1}, 0)))'
        row[45] = f'=IFNA(INDEX(lab_data!N:N, MATCH(1, (lab_data!A:A = B{last_row}) * (lab_data!E:E = MAX(FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E <= F{last_row}) * (lab_data!E:E >= F{last_row} - 30) * (lab_data!N:N <> "")))) * (lab_data!N:N <> ""), 0)), "")'
        row[46] = f'=IF(OR(AS{last_row}="",AV{last_row}=""),"",IF(AV{last_row}=0,0,AV{last_row}/DATEVALUE(INDEX(lab_data!$E:$E,MATCH(1,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E=MAX(FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})*(lab_data!N:N<>""))))*(lab_data!N:N<>""),0))-XLOOKUP(AS{last_row},FILTER(lab_data!N:N,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),FILTER(lab_data!$E:$E,(lab_data!$A:$A=$B{last_row})*(lab_data!$E:$E<=$F{last_row})),"Not found"))))'
        row[47] = f'=IF(AT{last_row}="", "", AT{last_row} - AS{last_row})'

        # nephrotoxins藥物使用
        row[48] = int(data[10])
        row[49] = int(data[11])
        row[50] = int(data[12])
        row[51] = int(data[13])
        row[52] = int(data[14])
        row[53] = int(data[15])
        row[54] = int(data[16])
        row[55] = int(data[17])
        row[56] = int(data[18])
        row[57] = f'=SUM(AW{last_row}:BE{last_row})'
        
        # post SCr和eGFR查找
        row[58] = f'=IFNA(IF(MAX(FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 89)))=0, "", MAX(FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 89)))), "")'
        row[59] = f'=IF(AZ{last_row}="","",TO_DATE(XLOOKUP(AZ{last_row}, FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 89)), FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 89)), "Not found")))'
        row[60] = f'=IF(AW{last_row}="", "", IF(D{last_row}=0, IF(AW{last_row}<=0.7, 141*((AW{last_row}/0.7)^-0.329)*0.993^E{last_row}*1.018, 141*((AW{last_row}/0.7)^-1.209)*0.993^E{last_row}*1.018), IF(AW{last_row}<=0.9, 141*((AW{last_row}/0.9)^-0.411)*0.993^E{last_row}, 141*((AW{last_row}/0.9)^-1.209)*0.993^E{last_row})))'
        row[61] = f'=IFNA(IF(MAX(FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 14)))=0, "", MAX(FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 14)))), "")'
        row[62] = f'=IF(AZ{last_row}="","",TO_DATE(XLOOKUP(AZ{last_row}, FILTER(lab_data!H:H, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 14)), FILTER(lab_data!E:E, (lab_data!A:A = B{last_row}) * (lab_data!E:E > F{last_row}) * (lab_data!E:E <= F{last_row} + 14)), "Not found")))'
        row[63] = f'=IF(AZ{last_row}="", "", IF(D{last_row}=0, IF(AZ{last_row}<=0.7, 141*((AZ{last_row}/0.7)^-0.329)*0.993^E{last_row}*1.018, 141*((AZ{last_row}/0.7)^-1.209)*0.993^E{last_row}*1.018), IF(AZ{last_row}<=0.9, 141*((AZ{last_row}/0.9)^-0.411)*0.993^E{last_row}, 141*((AZ{last_row}/0.9)^-1.209)*0.993^E{last_row})))'
        
        # AKI, AKD判定
        row[65] = f'=IF(AZ{last_row}="", 0, IF(D{last_row}=1,IF(R{last_row}>=1.3,IF(OR(AZ{last_row}/Q{last_row}>=1.5, AZ{last_row}/R{last_row}>=1.5), 1, 0),IF(OR(AZ{last_row}/Q{last_row}>=1.5, AZ{last_row}/R{last_row}>=1.5, AZ{last_row}/1.3>=1.5), 1, 0)),IF(R{last_row}>=1.1,IF(OR(AZ{last_row}/Q{last_row}>=1.5, AZ{last_row}/R{last_row}>=1.5), 1, 0),IF(OR(AZ{last_row}/Q{last_row}>=1.5, AZ{last_row}/R{last_row}>=1.5, AZ{last_row}/1.1>=1.5), 1, 0))))'
        row[66] = f'=IF(AW{last_row}="", 0, IF(V{last_row}<60, IF(OR(AW{last_row}/R{last_row}>=1.5,AW{last_row}/Q{last_row}>=1.5, AY{last_row}/V{last_row}<0.65,AY{last_row}/U{last_row}<0.65, BD{last_row}=1), 1, 0), IF(OR(AW{last_row}/R{last_row}>=1.5,AW{last_row}/Q{last_row}>=1.5, AY{last_row}/V{last_row}<0.65,AY{last_row}/U{last_row}<0.65, BD{last_row}=1, AY{last_row}<60), 1, 0)))'

        
        # AKI_history判定
        # 取得目前病人 ID 和給藥日期
        current_id = data[0]
        current_date = data[4]
        checkbox_checked = data[9]
        has_aki_history = False
        for r in reversed(all_rows[1:]):  # 從最新資料往回找
            if r[1] == current_id and r[5] < current_date and r[65] == "1":  # 注意：從 Google Sheet 抓下來是字串
                has_aki_history = True
                break
        if data[9] or has_aki_history : 
            row[64] = 1
        else :
            row[64] = 0  # UI 有勾 or 過去有 AKI 就是 1
        
        return row

    elif sheet_name == "lab_data":
        sheet = client.open("web data").worksheet("lab_data")
        last_row = len(sheet.get_all_values()) + 1
        row = ["" for _ in range(14)]  
        
        row[0], row[3], row[4] = data[0], data[1], data[2]
        row[6], row[7], row[11], row[12], row[13] = data[3], data[4], data[5], data[6], data[7]
        row[1] = f'=IFERROR(VLOOKUP(A{last_row}, INDIRECT("chemo_data!B:D"), 3, FALSE), "")'  # 查找性别
        row[2] = f'=IFERROR(VLOOKUP(A{last_row}, INDIRECT("chemo_data!B:E"), 4, FALSE), "")'  # 查找年紀
        # F 列: 如果 G (BUN) 有值，則填入 G，否則找最近的 BUN
        row[5] = f'=IF(G{last_row}<>"", G{last_row}, IF(ROW()=2, "", IFERROR(INDEX(G$2:G{last_row-1}, MAX(IF(A$2:A{last_row-1}=A{last_row}, ROW(A$2:A{last_row-1})-1, 0))), "")))'

        # I 列: 如果 H (Scr) 為空則為空，否則 F / H
        row[8] = f'=IF(OR(H{last_row}="", F{last_row}=""), "", F{last_row} / H{last_row})'
        # J 列: eGFR 计算
        row[9] = f'=IF(B{last_row}=0, IF(H{last_row}<=0.7, 141*((H{last_row}/0.7)^-0.329)*0.993^C{last_row}*1.018, 141*((H{last_row}/0.7)^-1.209)*0.993^H{last_row}*1.018), IF(H{last_row}<=0.9, 141*((H{last_row}/0.9)^-0.411)*0.993^C{last_row}, 141*((H{last_row}/0.9)^-1.209)*0.993^C{last_row}))'
        # K 列: CrCl 计算
        row[10] = f'=IF(B{last_row}=0, ((140 - C{last_row}) * D{last_row}) / (H{last_row} * 72) * 0.85, ((140 - C{last_row}) * D{last_row}) / (H{last_row} * 72))'

        sheet.append_row(row, value_input_option="USER_ENTERED")

# === 圖片 ===
st.image(
    "https://raw.githubusercontent.com/ChanWeiKai0118/AKD/main/AKI_AKD_prediction.jpg",
    width=800
)

# === 備註欄 ===
st.markdown(
    """
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #e6e9ef;">
        <h2 style="color: #333; text-align: center;">Definitions & Setting</h2>
        <p><strong>14-day AKI definition: (CTCAE 5.0)</strong></p>
        <ol>
            <li>An increase of SCr levels by >1.5 folds over baseline or latest SCr</li>
            <li>An increase of SCr levels by >1.5 folds over the upper limit of normal value (ULN)</li>
            <p style="margin-left: 20px;">(ULN: male 1.3 mg/dL, female 1.1 mg/dL)</p>
        </ol>
        <p><strong>89-day AKD definition: (ADQI 2016)</strong></p>
        <ol>
            <li>AKI</li>
            <li>eGFR drops to < 60 post chemotherapy (eGFR>60 before chemotherapy)</li>
            <li>eGFR decrease by > 35% over baseline or latest eGFR</li>
            <li>SCr increase by > 50% over baseline or latest SCr</li>
        </ol>
        <p><strong>※ Others</strong></p>
        <ul>
            <li>eGFR is calculated by CKD-EPI</li>
            <li>Baseline SCr : the latest SCr within 30 days before the first cycle</li>
            <li>Latest SCr : the latest SCr within 30 days before the current cycle</li>
        </ul>
        <p><strong>※ Model probability grading (for reference)</strong></p>
        <p><strong>AKD probability:</strong></p>
        <ul>
            <li><span style="color:green;">Very Low:</span> 0% ~ 16.7%</li>
            <li><span style="color:green;">Low:</span> 16.7% ~ 21.1%</li>
            <li><span style="color:orange;">Average:</span> 21.1% ~ 27.5%</li>
            <li><span style="color:red;">High:</span> 27.5% ~ 49.0%</li>
            <li><span style="color:red;">Very High:</span> 49.0% ~ 100%</li>
        </ul>
        <p><strong>AKI probability:</strong></p>
        <ul>
            <li><span style="color:green;">Very Low:</span> 0% ~ 0.4%</li>
            <li><span style="color:green;">Low:</span> 0.4% ~ 4.8%</li>
            <li><span style="color:orange;">Average:</span> 4.8% ~ 13.3%</li>
            <li><span style="color:red;">High:</span> 13.3% ~ 26.1%</li>
            <li><span style="color:red;">Very High:</span> 26.1% ~ 100%</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
# === Google sheet ===
# ---google sheet超連結---
sheet_url = "https://docs.google.com/spreadsheets/d/1G-o0659UDZQp2_CFEzty8mI0VXXYWzA0rc7v-Uz1ccc/edit?gid=0#gid=0"
st.markdown(f"[👉 Open Google Sheet]({sheet_url})", unsafe_allow_html=True)

# ---直接顯示google sheet---
sheet_url = "https://docs.google.com/spreadsheets/d/1G-o0659UDZQp2_CFEzty8mI0VXXYWzA0rc7v-Uz1ccc/edit?gid=0#gid=0"
st.components.v1.iframe(sheet_url, width=1000, height=600)

# === 第一個 Streamlit UI (檢驗數據) ===
st.markdown(
    """
    <div style="background-color: #d4f4dd; padding: 10px; border-radius: 8px;">
        <h1 style="color: black; text-align: center;">Laboratory Data Entry</h1>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.radio("Select mode", options=["Input data mode", "Check data mode"], horizontal=True)
# 輸入模式
if mode == "Input data mode":
    st.subheader("🔮 Input data Mode")
    col3, col4 = st.columns(2)
    
    with col3:
        lab_number = st.text_input("Patient ID (lab data)")
        weight_lab = st.number_input("Weight (kg) ", min_value=0.0, format="%.1f")
        lab_date = st.date_input("Date", datetime.date.today())
    
    with col4:
        bun = st.number_input("BUN (mg/dL)", min_value=0.0, value=None)
        scr = st.number_input("Scr (mg/dL)", min_value=0.00, format="%.2f", value=None)
        hgb = st.number_input("Hgb (g/dL)", min_value=0.0, format="%.1f", value=None)
        sodium = st.number_input("Sodium (mmol/L)", min_value=0, value=None)
        potassium = st.number_input("Potassium (mmol/L)", min_value=0, value=None)
    
    if st.button("Submit Lab Data"):
        lab_date_str = lab_date.strftime("%Y/%m/%d")
        lab_data_list = [lab_number, weight_lab, lab_date_str, bun or "", scr or "", hgb or "", sodium or "", potassium or ""]
        save_to_gsheet(lab_data_list, "lab_data")
        st.success("✅ Laboratory data submitted successfully!")
        # 👉 顯示剛剛輸入的資料
        lab_df = pd.DataFrame([lab_data_list], columns=['Number', 'Weight', 'Date','Scr','BUN','Hb','Na','K'])
        st.subheader("🧾 Submitted Data")
        st.dataframe(lab_df)
# -----------------------------
# 預覽模式
elif mode == "Check data mode":
    st.subheader("🗂️ Check Data Mode")
    number_check = st.text_input("Input patient ID", key="check_id")
    if st.button("Check Lab Data"):
        if number_check:
            try:
                number_check = str(number_check).zfill(8)  # 強制補滿8位數
                client = get_gsheet_client()
                sheet = client.open("web data").worksheet("lab_data")
                all_data = sheet.get_all_records()
                df = pd.DataFrame(all_data)
                preview_cols = ['Number', 'Weight', 'Date','Scr','BUN','Hb','Na','K']
                filtered_df = df[preview_cols]
                # 👉 將 Number 欄位全部轉成補滿8位的字串格式
                filtered_df['Number'] = filtered_df['Number'].astype(str).str.zfill(8)
                filtered_df = filtered_df[filtered_df['Number'] == number_check]
                
                if not filtered_df.empty:
                    st.subheader(f"Patient information（ID: {number_check}）")
                    st.dataframe(filtered_df)
                else:
                    st.info("❗ The patient has no lab data")
            except Exception as e:
                st.error(f"Something wrong when loading Google Sheet ：{e}")
        else:
            st.warning("Please enter patient ID")



# =======================
# AKD Prediction Function
# =======================
def run_prediction_AKD(selected_rows):
    # AKD columns
    # 加入'carb_dose','dose_percentage','cis_cycle','cis_cum_dose'方便後續做劑量調整
    target_columns = [
        'id_no', 'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 
         'cis_dose','carb_dose', 'latest_hemoglobin', 'latest_egfr', 'baseline_hemoglobin', 
         'baseline_bun', 'baseline_bun/scr', 'baseline_crcl', 'baseline_sodium', 
         'baseline_potassium', 'carb_cum_dose', 'average_cis_cum_dose', 
         'hemoglobin_change', 'crcl_change', 'potassium_change', 'crcl_slope', 
         'aki_history','dose_percentage','cis_cycle','cis_cum_dose'
    ]
    cols_for_preprocessing = [
        'id_no', 'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 
         'cis_dose', 'latest_hemoglobin', 'latest_egfr', 'baseline_hemoglobin', 
         'baseline_bun', 'baseline_bun/scr', 'baseline_crcl', 'baseline_sodium', 
         'baseline_potassium', 'carb_cum_dose', 'average_cis_cum_dose', 
         'hemoglobin_change', 'crcl_change', 'potassium_change', 'crcl_slope', 
         'aki_history','akd'
    ]
    selected_features = [
        'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 
         'cis_dose', 'latest_hemoglobin', 'latest_egfr', 'baseline_hemoglobin', 
         'baseline_bun', 'baseline_bun/scr', 'baseline_crcl', 'baseline_sodium', 
         'baseline_potassium', 'carb_cum_dose', 'average_cis_cum_dose', 
         'hemoglobin_change', 'crcl_change', 'potassium_change', 'crcl_slope', 
         'aki_history'
    ]
    # === 讀取 Google Sheet 資料 ===
    input_data = selected_rows[target_columns].apply(pd.to_numeric, errors='coerce')
    input_data.reset_index(drop=True, inplace=True)
    input_data.loc[input_data.index[-1], 'akd'] = 0  # outcome column
    
    # 取得原本資料是用cisplatin or carboplatin
    last_row_index = input_data.index[-1]
    original_cis_dose = input_data.loc[last_row_index, 'cis_dose']
    original_carb_dose = input_data.loc[last_row_index, 'carb_dose']
    dose_percentage = input_data.loc[last_row_index, 'dose_percentage']
    if pd.notna(original_cis_dose) and original_cis_dose > 0:
        dose_type = 'Cisplatin'
    elif pd.notna(original_carb_dose) and original_carb_dose > 0:
        dose_type = 'Carboplatin'
    else:
        dose_type = None

    # 儲存預測那筆資料的dose percentage
    dose_percentage = input_data.loc[last_row_index, 'dose_percentage']
    # 在傳入 preprocessing 前，移除 'carb_dose','dose_percentage','cis_cycle'
    input_data_pred = input_data.drop(columns=['carb_dose','dose_percentage','cis_cycle','cis_cum_dose'])

    # preprocessing
    normalizer = get_scaler()
    miceforest = get_imputer()
    X_test, y_test = preprocessing(
        data=input_data_pred,
        scaler=normalizer,
        imputer=miceforest,
        cols_for_preprocessing=cols_for_preprocessing,
        groupby_col='id_no', 
        selected_features=selected_features,
        outcome='akd',
        maxlen=6
    )
    model = get_model()

    # 过滤掉 padding 数据
    y_prob = model.predict(X_test).squeeze().flatten()
    sample_weight = (y_test != -1).astype(float).flatten()
    valid_indices = sample_weight > 0
    flat_prob = y_prob[valid_indices]
    last_prob = flat_prob[-1] * 100

    
    
    # 針對不同百分比劑量進行預測
    dose_adjustments = [100, 90, 80, 70]
    prediction_results = {}
    for percentage in dose_adjustments:
        input_data_modified = input_data.copy()
        if dose_type == 'Cisplatin':
            new_cis_dose = original_cis_dose / dose_percentage * percentage
            input_data_modified.loc[last_row_index, 'cis_dose'] = new_cis_dose
            prev = input_data_modified.loc[last_row_index - 1, 'cis_cum_dose'] if last_row_index > 0 else 0
            input_data_modified.loc[last_row_index, 'cis_cum_dose'] = prev + new_cis_dose
            cis_cycle = input_data_modified.loc[last_row_index, 'cis_cycle']
            input_data_modified.loc[last_row_index, 'average_cis_cum_dose'] = input_data_modified.loc[last_row_index, 'cis_cum_dose'] / cis_cycle
        elif dose_type == 'Carboplatin':
            new_carb_dose = original_carb_dose / dose_percentage * percentage
            input_data_modified.loc[last_row_index, 'carb_dose'] = new_carb_dose
            prev = input_data_modified.loc[last_row_index - 1, 'carb_cum_dose'] if last_row_index > 0 else 0
            input_data_modified.loc[last_row_index, 'carb_cum_dose'] = prev + new_carb_dose

        input_data_modified_pred = input_data_modified.drop(columns=['carb_dose','dose_percentage','cis_cycle','cis_cum_dose'])
        X_test_dose, y_test_dose = preprocessing(
            data=input_data_modified_pred,
            scaler=normalizer,
            imputer=miceforest,
            cols_for_preprocessing=cols_for_preprocessing,
            groupby_col='id_no',
            selected_features=selected_features,
            outcome='akd',
            maxlen=6
        )
        y_prob_dose = model.predict(X_test_dose).squeeze().flatten()
        valid_indices = (y_test_dose != -1).astype(bool).flatten()
        flat_prob_dose = y_prob_dose[valid_indices]
        prediction_results[f'{percentage}%'] = flat_prob_dose[-1] * 100

    return last_prob, prediction_results,dose_percentage


# =======================
# AKI Prediction Function
# =======================
def run_prediction_AKI(selected_rows):
    #AKI columns
    # 加入'carb_dose','dose_percentage','cis_cycle'方便後續做劑量調整
    target_columns = [
        'id_no', 'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 'cis_dose', 'carb_dose',
        'latest_hemoglobin', 'latest_bun/scr', 'latest_egfr', 'baseline_hemoglobin',
        'baseline_bun/scr', 'baseline_crcl', 'cis_cum_dose', 'carb_cum_dose',
        'average_cis_cum_dose', 'bun_change', 'bun/scr_change', 'crcl_change',
        'potassium_change', 'crcl_slope', 'aki_history','dose_percentage','cis_cycle'
    ]
    cols_for_preprocessing = [
        'id_no', 'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 'cis_dose', 
        'latest_hemoglobin', 'latest_bun/scr', 'latest_egfr', 'baseline_hemoglobin',
        'baseline_bun/scr', 'baseline_crcl', 'cis_cum_dose', 'carb_cum_dose',
        'average_cis_cum_dose', 'bun_change', 'bun/scr_change', 'crcl_change',
        'potassium_change', 'crcl_slope', 'aki_history','aki'
    ]
    selected_features = [
        'age', 'weight', 'number_of_nephrotoxins', 'treatment_duration', 'cis_dose', 
        'latest_hemoglobin', 'latest_bun/scr', 'latest_egfr', 'baseline_hemoglobin',
        'baseline_bun/scr', 'baseline_crcl', 'cis_cum_dose', 'carb_cum_dose',
        'average_cis_cum_dose', 'bun_change', 'bun/scr_change', 'crcl_change',
        'potassium_change', 'crcl_slope', 'aki_history'
    ]
    
    # === 讀取 Google Sheet 資料 ===
    input_data = selected_rows[target_columns].apply(pd.to_numeric, errors='coerce')
    input_data.reset_index(drop=True, inplace=True)
    input_data.loc[input_data.index[-1], 'aki'] = 0
    
    # 取得原本資料是用cisplatin or carboplatin
    last_row_index = input_data.index[-1]
    original_cis_dose = input_data.loc[last_row_index, 'cis_dose']
    original_carb_dose = input_data.loc[last_row_index, 'carb_dose']
    dose_percentage = input_data.loc[last_row_index, 'dose_percentage']
    if pd.notna(original_cis_dose) and original_cis_dose > 0:
        dose_type = 'Cisplatin'
    elif pd.notna(original_carb_dose) and original_carb_dose > 0:
        dose_type = 'Carboplatin'
    else:
        dose_type = None

    # 儲存預測那筆資料的dose percentage
    dose_percentage = input_data.loc[last_row_index, 'dose_percentage']
    # 在傳入 preprocessing 前，移除 'carb_dose','dose_percentage','cis_cycle'
    input_data_pred = input_data.drop(columns=['carb_dose','dose_percentage','cis_cycle'])

    # Preprocess
    normalizer = get_aki_scaler()
    miceforest = get_aki_imputer()
    X_test, y_test = preprocessing(
        data=input_data_pred,
        scaler=normalizer,
        imputer=miceforest,
        cols_for_preprocessing=cols_for_preprocessing,
        groupby_col='id_no',  
        selected_features=selected_features,
        outcome='aki',
        maxlen=6
    )
    model = get_aki_model()
    
    # 过滤掉 padding 数据
    y_prob = model.predict(X_test).squeeze().flatten()
    sample_weight = (y_test != -1).astype(float).flatten()
    valid_indices = sample_weight > 0
    flat_prob = y_prob[valid_indices]
    last_prob = flat_prob[-1] * 100

    # 針對不同百分比劑量進行預測
    dose_adjustments = [100, 90, 80, 70]
    prediction_results = {}
    for percentage in dose_adjustments:
        input_data_modified = input_data.copy()
        if dose_type == 'Cisplatin':
            new_cis_dose = original_cis_dose / dose_percentage * percentage
            input_data_modified.loc[last_row_index, 'cis_dose'] = new_cis_dose
            prev = input_data_modified.loc[last_row_index - 1, 'cis_cum_dose'] if last_row_index > 0 else 0
            input_data_modified.loc[last_row_index, 'cis_cum_dose'] = prev + new_cis_dose
            cis_cycle = input_data_modified.loc[last_row_index, 'cis_cycle']
            input_data_modified.loc[last_row_index, 'average_cis_cum_dose'] = input_data_modified.loc[last_row_index, 'cis_cum_dose'] / cis_cycle
        elif dose_type == 'Carboplatin':
            new_carb_dose = original_carb_dose / dose_percentage * percentage
            input_data_modified.loc[last_row_index, 'carb_dose'] = new_carb_dose
            prev = input_data_modified.loc[last_row_index - 1, 'carb_cum_dose'] if last_row_index > 0 else 0
            input_data_modified.loc[last_row_index, 'carb_cum_dose'] = prev + new_carb_dose

        input_data_modified_pred = input_data_modified.drop(columns=['carb_dose','dose_percentage','cis_cycle'])
        X_test_dose, y_test_dose = preprocessing(
            data=input_data_modified_pred,
            scaler=normalizer,
            imputer=miceforest,
            cols_for_preprocessing=cols_for_preprocessing,
            groupby_col='id_no',
            selected_features=selected_features,
            outcome='aki',
            maxlen=6
        )
        y_prob_dose = model.predict(X_test_dose).squeeze().flatten()
        valid_indices = (y_test_dose != -1).astype(bool).flatten()
        flat_prob_dose = y_prob_dose[valid_indices]
        prediction_results[f'{percentage}%'] = flat_prob_dose[-1] * 100

    return last_prob, prediction_results, dose_percentage

def get_aki_color(prob):
    if prob <= 0.4:
        return "green"   # Very Low
    elif prob <= 4.8:
        return "green"   # Low
    elif prob <= 13.3:
        return "orange"  # Average
    elif prob <= 26.1:
        return "red"     # High
    else:
        return "red"     # Very High

def get_akd_color(prob):
    if prob <= 16.7:
        return "green"   # Very Low
    elif prob <= 21.1:
        return "green"   # Low
    elif prob <= 27.5:
        return "orange"  # Average
    elif prob <= 49.0:
        return "red"     # High
    else:
        return "red"     # Very High


# === 第二個 Streamlit UI ===
st.markdown(
    """
    <div style="background-color: #FFFFE0; padding: 10px; border-radius: 8px;">
        <h1 style="color: black; text-align: center;">Chemotherapy Data Entry</h1>
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.radio("Select mode", options=["Input mode", "Check mode","Prediction mode"], horizontal=True)

# 輸入模式
if mode == "Input mode":
    st.subheader("🔮 Input Mode")
    col1, col2, col3 = st.columns(3)

    with col1:
        number = st.text_input("Patient ID (chemotherapy data)", key="predict_id")
        weight = st.number_input("Weight (kg)", min_value=0.0, format="%.1f")
        gender = st.selectbox("Gender", ["Male", "Female"])
        gender_value = 1 if gender == "Male" else 0
        age = st.number_input("Age (years)", min_value=0)
        aki_history = st.checkbox("AKI History (Check if Yes)")

    with col2:
        treatment_date = st.date_input("Treatment Date", datetime.date.today())
        cycle_no = st.number_input("Cycle Number", min_value=1)
        cis_dose = st.number_input("Cisplatin Dose (mg)", min_value=0.0, format="%.1f")
        carb_dose = st.number_input("Carboplatin Dose (mg)", min_value=0.0, format="%.1f")
        dose_percentage = st.number_input("Dose percentage (%)", min_value=0, max_value=100)
    
    with col3:
        st.markdown("**Nephrotoxin Medications**")
        acei_arb = st.checkbox("ACEI/ARB")
        acetaminophen = st.checkbox("Acetaminophen")
        diuretics = st.checkbox("Diuretics")
        h2_blocker = st.checkbox("H2-blocker")
        nsaids = st.checkbox("NSAIDs")
        beta_lactam = st.checkbox("Beta-lactam")
        ppi = st.checkbox("PPI")
        contrast_media = st.checkbox("Contrast media")
        others = st.checkbox("Others (Allopurinol, aminoglycosides, vancomycin, antivirals, fluoroquinolone, colistin)")

    
    if st.button("Submit Chemo Data"):
        treatment_date_str = treatment_date.strftime("%Y/%m/%d")
        number = str(number).zfill(8)  # 強制補滿8位數
        chemo_data_list = [
            number, gender_value, weight, age, 
            treatment_date_str, cycle_no, dose_percentage, cis_dose, carb_dose, aki_history ,
            acei_arb, acetaminophen, diuretics, h2_blocker, nsaids, beta_lactam, ppi, contrast_media, others
        ]
    
        # 回傳資料行、AKI 判定結果、病人 ID
        row_to_write = save_to_gsheet(chemo_data_list, "chemo_data")

        # 這裡才送出資料
        sheet = get_gsheet_client().open("web data").worksheet("chemo_data")
        sheet.append_row(row_to_write, value_input_option="USER_ENTERED")
    
        st.success("✅ Data submitted successfully!")
        # 👉 顯示剛剛輸入的資料
        chemo_df = pd.DataFrame([chemo_data_list], columns=['Number','Gender','Weight', 'Age','Date','Cycle','Dose percentage(%)','Cisplatin dose','Carboplatin dose','AKI history',
                                                            'acei_arb','acetaminophen','diuretics','h2_blocker','nsaids','beta_lactam','ppi','contrast','other_nephrotoxin'])
        st.subheader("🧾 Submitted Data")
        st.dataframe(chemo_df)
        
        
# -----------------------------
# 預覽模式
elif mode == "Check mode":
    st.subheader("🗂️ Check Mode")
    number_preview = st.text_input("Input patient ID", key="preview_id")
    if st.button("Check Chemo Data"):
        number_preview = str(number_preview).zfill(8)  # 強制補滿8位數
        if number_preview:
            try:
                client = get_gsheet_client()
                sheet = client.open("web data").worksheet("chemo_data")
                all_data = sheet.get_all_records()
                df = pd.DataFrame(all_data)
                preview_cols = ['Number', 'weight', 'sex_male', 'age', 'Index_date 1(dose)', 'cis_cycle', 'carb_cycle', 'cis_dose','carb_dose','aki_history']
                filtered_df = df[preview_cols]
                # 👉 將 Number 欄位全部轉成補滿8位的字串格式
                filtered_df['Number'] = filtered_df['Number'].astype(str).str.zfill(8)
                filtered_df = filtered_df[filtered_df['Number'] == number_preview]
                
                if not filtered_df.empty:
                    st.subheader(f"Patient information（ID: {number_preview}）")
                    st.dataframe(filtered_df)
                else:
                    st.info("❗ The patient has no chemotherapy data")
            except Exception as e:
                st.error(f"Something wrong when loading Google Sheet ：{e}")
        else:
            st.warning("Please enter patient ID")
# -----------------------------
# 預測模式

elif mode == "Prediction mode":
    st.subheader("🔮 AKD & AKI prediction")    
    input_number = st.text_input("Enter Patient ID (Number):")
    input_date = st.date_input("Treatment Date", datetime.date.today())
    input_date_str = input_date.strftime("%Y/%m/%d")

    if st.button("Run Prediction"):
        if input_number and input_date_str:
            try:
                client = get_gsheet_client()
                sheet = client.open("web data").worksheet("chemo_data")
                raw_values = sheet.get_all_values()
                headers = raw_values[0]
                data = raw_values[1:]
                df = pd.DataFrame(data, columns=headers)
                
                # === 找到該筆預測的 row ===
                df_patient = df[df['Number'] == input_number]
                df_patient = df_patient.sort_values(by='Index_date 1(dose)')

                # 找到最接近輸入日期的 row
                selected_row = df_patient[df_patient['Index_date 1(dose)'] == input_date_str]

                if selected_row.empty:
                    st.warning("No exact match found for this date. Please check again.")
                else:
                    target_index = selected_row.index[0]
                    
                    # 只保留相同 id_no 的 row
                    id_no_value = selected_row.iloc[0]['id_no']
                    df_patient = df_patient[df_patient['id_no'] == id_no_value]
    
                    # 取輸入日期當下 row 之前的最近 6 筆
                    selected_rows = df_patient.loc[:target_index].tail(6)
                    
                    # 顯示預測用資料
                    st.subheader("Data for Prediction")
                    st.dataframe(selected_rows)

                    # Run AKD
                    st.markdown("## 🧮 AKD Prediction")
                    akd_prob, akd_results,dose_percentage= run_prediction_AKD(selected_rows)
                    st.markdown(f"### Predicted AKD Risk: <span style='color:{get_akd_color(akd_prob)};'>{akd_prob:.4f}%</span> (dose at {dose_percentage}%)",unsafe_allow_html=True)
                    for k, v in akd_results.items():
                        st.info(f"{k} dose → Predicted AKD Risk: **{v:.4f}%**")

                    # Run AKI
                    st.markdown("## 🧮 AKI Prediction")
                    aki_prob, aki_results,dose_percentage = run_prediction_AKI(selected_rows)
                    st.markdown(f"### Predicted AKI Risk: <span style='color:{get_aki_color(aki_prob)};'>{aki_prob:.2f}%</span> (dose at {dose_percentage}%)",unsafe_allow_html=True)
                    for k, v in aki_results.items():
                        st.info(f"{k} dose → Predicted AKI Risk: **{v:.2f}%**")

            except Exception as e:
                import traceback
                st.error(f"Error processing your request: {str(e)}")
                st.text(traceback.format_exc())












































