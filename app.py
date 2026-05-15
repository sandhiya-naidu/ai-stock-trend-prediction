import streamlit as st
from auth import login, register
import yfinance as yf
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Trend Prediction System",
    page_icon="📈",
    layout="wide"
)

# ---------------- SESSION ----------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# ---------------- LOGIN BACKGROUND FUNCTION ----------------
def login_background():

        with open("stock_bg.png", "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(
        f"""
        <style>

        .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        }}

        h1,h2,h3 {{
        color:white !important;
        }}

        label {{
        color:white !important;
        }}

        button[data-baseweb="tab"] {{
        color:white !important;
        font-weight:bold !important;
        }}

        button[aria-selected="true"] {{
        background-color:#00cc99 !important;
        color:white !important;
        }}

        button[aria-selected="false"] {{
        color:white !important;
        }}

        </style>
        """, unsafe_allow_html=True)

# ---------------- DARK UI (MAIN PAGE) ----------------
def main_background():
    st.markdown("""
    <style>

    .stApp{
    background-color:#000000;
    color:#ffffff;
    }

    h1,h2,h3{
    color:#ffffff !important;
    font-weight:800 !important;
    }

    label{
    color:#ffffff !important;
    font-size:16px !important;
    font-weight:600 !important;
    }

    input{
    color:#ffffff !important;
    font-weight:bold !important;
    }

    div[data-baseweb="select"]{
    color:#ffffff !important;
    font-weight:bold !important;
    }


    div[data-testid="stDateInput"] input{
    background-color:white !important;
    color:black !important;
    font-weight:bold !important;
    }

    .metric-number{
    color:#00ffcc !important;
    font-size:32px;
    font-weight:800;
    }

    .up{
    color:#00ff66;
    font-size:28px;
    font-weight:bold;
    }

    .down{
    color:#ff4d4d;
    font-size:28px;
    font-weight:bold;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGIN SCREEN ----------------
if not st.session_state["authenticated"]:

    login_background()

    st.title(" Stock Trend Prediction System")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login()

    with tab2:
        register()

    st.stop()

# ---------------- MAIN PAGE STYLE ----------------
main_background()

# ---------------- HEADER ----------------
st.title(" Stock Trend Prediction System")
st.caption("AI-based market direction assistant")

# ---------------- NIFTY 50 LIST ----------------
nifty_50 = {
    "Adani Enterprises":"ADANIENT.NS","Adani Ports":"ADANIPORTS.NS","Apollo Hospitals":"APOLLOHOSP.NS",
    "Asian Paints":"ASIANPAINT.NS","Axis Bank":"AXISBANK.NS","Bajaj Auto":"BAJAJ-AUTO.NS",
    "Bajaj Finance":"BAJFINANCE.NS","Bajaj Finserv":"BAJAJFINSV.NS","Bharti Airtel":"BHARTIARTL.NS",
    "BPCL":"BPCL.NS","Britannia":"BRITANNIA.NS","Cipla":"CIPLA.NS","Coal India":"COALINDIA.NS",
    "Divis Labs":"DIVISLAB.NS","Dr Reddy":"DRREDDY.NS","Eicher Motors":"EICHERMOT.NS",
    "Grasim":"GRASIM.NS","HCL Tech":"HCLTECH.NS","HDFC Bank":"HDFCBANK.NS","HDFC Life":"HDFCLIFE.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS","Hindalco":"HINDALCO.NS","HUL":"HINDUNILVR.NS",
    "ICICI Bank":"ICICIBANK.NS","ITC":"ITC.NS","IndusInd Bank":"INDUSINDBK.NS","Infosys":"INFY.NS",
    "JSW Steel":"JSWSTEEL.NS","Kotak Bank":"KOTAKBANK.NS","LT":"LT.NS","Maruti":"MARUTI.NS",
    "Nestle":"NESTLEIND.NS","NTPC":"NTPC.NS","ONGC":"ONGC.NS","Power Grid":"POWERGRID.NS",
    "Reliance":"RELIANCE.NS","SBI":"SBIN.NS","SBILife":"SBILIFE.NS","Sun Pharma":"SUNPHARMA.NS",
    "TCS":"TCS.NS","Tata Motors":"TATAMOTORS.NS","Tata Steel":"TATASTEEL.NS","Tech Mahindra":"TECHM.NS",
    "Titan":"TITAN.NS","UPL":"UPL.NS","Ultratech Cement":"ULTRACEMCO.NS","Wipro":"WIPRO.NS"
}

# ---------------- INPUTS ----------------
col1,col2,col3 = st.columns(3)

with col1:
    company = st.selectbox("Select NIFTY 50 Stock", list(nifty_50.keys()))
    symbol = nifty_50[company]

with col2:
    start_date = st.date_input("From", datetime(2022,1,1))

with col3:
    end_date = st.date_input("To", datetime.today())

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=False)
def load_data(sym,start,end):

    try:
        df = yf.download(sym,start=start,end=end,progress=False,auto_adjust=True)

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns,pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Close' not in df.columns:
            return pd.DataFrame()

        return df[['Close']].dropna()

    except:
        return pd.DataFrame()

data = load_data(symbol,start_date,end_date)

if data.empty:
    st.warning("⚠ Data not available for this stock and date range.")
    st.stop()

# ---------------- FEATURE ENGINEERING ----------------
data['Return'] = data['Close'].pct_change()
data['MA5'] = data['Close'].rolling(5).mean()
data['MA20'] = data['Close'].rolling(20).mean()
data.dropna(inplace=True)

data['Target'] = np.where(data['Close'].shift(-1) > data['Close'],1,0)

X = data[['Return','MA5','MA20']]
y = data['Target']

# ---------------- MODEL ----------------
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

model = RandomForestClassifier(n_estimators=200,max_depth=6,random_state=42)

model.fit(X_train,y_train)

# ---------------- ACCURACY ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)*100

# ---------------- PREDICTION ----------------
latest = X.iloc[-1:]

pred = model.predict(latest)[0]

confidence = model.predict_proba(latest).max()*100

trend,cls = ("UP ","up") if pred==1 else ("DOWN ","down")

# ---------------- METRICS ----------------
m1,m2,m3 = st.columns(3)

with m1:
    st.markdown(f"<div>Predicted Trend<br><span class='{cls}'>{trend}</span></div>",unsafe_allow_html=True)

with m2:
    st.markdown(f"<div>Model Confidence<br><span class='metric-number'>{confidence:.2f}%</span></div>",unsafe_allow_html=True)

with m3:
    st.markdown(f"<div>Test Accuracy<br><span class='metric-number'>{accuracy:.2f}%</span></div>",unsafe_allow_html=True)

# ---------------- CHART ----------------
st.markdown("### Stock Price Trend")

st.line_chart(data[['Close','MA5','MA20']])