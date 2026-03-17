import streamlit as st
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# ตั้งค่าหน้าแอป
st.set_page_config(page_title="Flu Forecast Thailand", layout="centered", initial_sidebar_state="collapsed")
warnings.filterwarnings("ignore")

# --- CSS ตกแต่ง (บังคับ Light Mode และสีกรมท่าทั้งหมด) ---
st.markdown("""
    <style>
    /* บังคับพื้นหลังหลักให้สว่าง */
    .stApp, .main, [data-testid="stAppViewContainer"] { 
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%) !important; 
        color: #1E3A8A !important;
    }
    
    /* บังคับตัวอักษรทุกชนิดให้เป็นสีกรมท่า (ป้องกัน Dark Mode เปลี่ยนสี) */
    p, span, li, h1, h2, h3, h4, div.stMarkdown, label {
        color: #1E3A8A !important; 
    }
    
    /* --- ส่วนเมนู Tabs --- */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center; 
        gap: 15px; 
        background-color: rgba(255, 255, 255, 0.7) !important; 
        padding: 10px; 
        border-radius: 20px; 
        margin: 0 auto; 
    }
    div[data-baseweb="tab-highlight"] { display: none; }

    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important; 
        padding: 10px 25px !important; 
        border-radius: 15px !important;
    }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span {
        font-size: 18px !important; 
        font-weight: 800 !important; 
    }
    
    /* Tab ที่ถูกเลือก */
    button[aria-selected="true"] {
        background-color: #FFFFFF !important; 
        box-shadow: 0 4px 10px rgba(30, 58, 138, 0.2) !important;
    }

    /* --- ส่วนข้อมูลโรค (Expander) ให้อ่านง่ายสุดๆ --- */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important; 
        border-radius: 10px !important; 
        border: 1px solid #a1c4fd !important;
    }
    .streamlit-expanderHeader p {
        font-weight: 900 !important;
        font-size: 18px !important;
    }
    /* พื้นหลังเนื้อหาด้านใน Expander */
    [data-testid="stExpanderDetails"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 15px !important;
    }

    /* --- วงกลมและตัวเลข --- */
    .glass-container {
        position: relative; width: 340px; height: 340px;
        margin: 20px auto; display: flex; align-items: center; justify-content: center;
    }
    .glass-circle {
        background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.4));
        border-radius: 50%; box-shadow: 0 15px 35px rgba(31, 38, 135, 0.2), inset 0 0 20px rgba(255, 255, 255, 0.9); 
        backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 1);
        width: 270px; height: 270px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; z-index: 10;
    }
    .week-text { font-size: 20px; font-weight: bold; margin-bottom: -5px; }
    .number-text { font-size: 50px; font-weight: 900; line-height: 1.1; text-shadow: 2px 2px 4px rgba(255,255,255,0.8); }
    .unit-text { font-size: 22px; font-weight: 700; margin-top: 5px; }
    
    /* ไวรัสลอยได้ */
    .virus-img { position: absolute; animation: float 6s ease-in-out infinite; z-index: 1; filter: drop-shadow(2px 5px 5px rgba(0,0,0,0.2)); }
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(10deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    .header-title { text-align: center; font-weight: 900; margin: 20px 0; font-size: 28px; text-shadow: 1px 1px 3px rgba(255,255,255,0.7); }
    </style>
    """, unsafe_allow_html=True)

virus_url = "https://cdn.pixabay.com/photo/2020/04/29/07/54/coronavirus-5107715_1280.png"

@st.cache_data
def load_and_predict():
    df_raw = pd.read_excel('data_flu.xlsx', engine="openpyxl")
    df = df_raw.rename(columns={
        'date_dt': 'Date', 'week': 'Week',
        'Patient rate per 100,000': 'Flu_Rate',
        'ILI cases per 100000': 'ILI_Rate'
    })
    
    df['Target_Rate'] = df['Flu_Rate'].rolling(window=3, center=True).mean().fillna(df['Flu_Rate'])
    df['ILI_Lag1'] = df['ILI_Rate'].shift(1)
    df_clean = df.dropna()

    y = df_clean['Target_Rate']
    exog = df_clean[['ILI_Lag1']]

    auto_model = pm.auto_arima(
        y, exogenous=exog, start_p=0, start_q=0, max_p=2, max_q=2,
        m=52, seasonal=True, D=1, trace=False, error_action='ignore',  
        suppress_warnings=True, stepwise=True
    )
    
    model = SARIMAX(y, exog=exog, order=auto_model.order, seasonal_order=auto_model.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    last_week_exog = exog.tail(1) 
    forecast = results.get_forecast(steps=1, exog=last_week_exog)
    pred_rate = forecast.predicted_mean.iloc[0]
    
    population = 66097304
    total_people = (pred_rate * population) / 100000
    
    last_week_num = int(df['Week'].iloc[-1])
    next_week_num = 1 if last_week_num >= 52 else last_week_num + 1
    
    return total_people, next_week_num

# --- การแสดงผล ---
tab1, tab2 = st.tabs(["หน้าหลัก", "ข้อมูลโรค"])

with tab1:
    try:
        total_people, next_week_num = load_and_predict()

        st.markdown("<div class='header-title'>คาดการณ์ผู้ป่วย<br>ไข้หวัดใหญ่สัปดาห์ถัดไป</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="glass-container">
                <img src="{virus_url}" class="virus-img" style="top:-5%; left:-5%; width:120px; animation-delay: 0s;">
                <img src="{virus_url}" class="virus-img" style="top:20%; right:-10%; width:80px; animation-delay: 1s;">
                <img src="{virus_url}" class="virus-img" style="bottom:5%; left:-10%; width:70px; animation-delay: 2s;">
                <img src="{virus_url}" class="virus-img" style="bottom:-10%; right:10%; width:100px; animation-delay: 1.5s;">
                <div class="glass-circle">
                    <span class="week-text">สัปดาห์ที่ {next_week_num}</span>
                    <span class="number-text">{int(total_people):,}</span>
                    <span class="unit-text">คน</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

with tab2:
    st.markdown("<div class='header-title'>ข้อมูลโรคไข้หวัดใหญ่</div>", unsafe_allow_html=True)
    
    with st.expander("อาการของผู้ป่วย", expanded=True):
        st.write("""
        * **ไข้สูงเฉียบพลัน** (มักสูงกว่า 38 องศาเซลเซียส)
        * **ปวดเมื่อยตามตัวมาก** โดยเฉพาะบริเวณหลัง แขน และขา
        * ปวดศีรษะ อ่อนเพลียอย่างรุนแรง (ลุกไม่ค่อยขึ้น)
        * ไอแห้ง เจ็บคอ มีน้ำมูกไหล
        """)
        
    with st.expander("สายพันธุ์โรคไข้หวัดใหญ่"):
        st.write("""
        เชื้อไวรัสไข้หวัดใหญ่ (Influenza virus) แบ่งออกเป็น 3 สายพันธุ์หลักในคน:
        * **สายพันธุ์ A:** พบได้บ่อยที่สุด มีความรุนแรง และสามารถกลายพันธุ์จนก่อให้เกิดการระบาดใหญ่ทั่วโลกได้ (เช่น H1N1, H3N2)
        * **สายพันธุ์ B:** มักทำให้เกิดการระบาดในระดับภูมิภาคในช่วงฤดูฝนและฤดูหนาว อาการมักรุนแรงน้อยกว่าสายพันธุ์ A 
        * **สายพันธุ์ C:** มักทำให้เกิดอาการป่วยเพียงเล็กน้อยคล้ายไข้หวัดธรรมดา และไม่ก่อให้เกิดการระบาดใหญ่
        """)

    with st.expander("วิธีการรักษา"):
        st.write("""
        * **การรักษาตามอาการ:** ทานยาลดไข้ (พาราเซตามอล) ยาแก้ไอ ยาลดน้ำมูก (หลีกเลี่ยงยาแอสไพรินในเด็ก)
        * **การพักผ่อน:** นอนหลับพักผ่อนให้เพียงพอ และดื่มน้ำสะอาดมากๆ เพื่อป้องกันภาวะขาดน้ำ
        * **ยาต้านไวรัส:** ในผู้ป่วยที่มีอาการรุนแรงหรือมีความเสี่ยงสูง แพทย์อาจพิจารณาให้ยาต้านไวรัส (เช่น Oseltamivir)
        * **พบแพทย์:** หากอาการไม่ดีขึ้นภายใน 3-5 วัน หรือมีอาการหอบเหนื่อย ควรรีบไปพบแพทย์ทันที
        """)
        
    with st.expander("🛡️ การป้องกัน"):
        st.write("""
        * **ฉีดวัคซีน:** แนะนำให้กลุ่มเสี่ยงและบุคคลทั่วไปฉีดวัคซีนป้องกันไข้หวัดใหญ่เป็นประจำทุกปี
        * **สวมหน้ากากอนามัย:** เมื่อต้องไปในที่ชุมชน หรือเมื่อตนเองมีอาการป่วย
        * **รักษาความสะอาด:** ล้างมือบ่อยๆ ด้วยสบู่และน้ำ หรือใช้เจลแอลกอฮอล์ หลีกเลี่ยงการจับใบหน้า
        """)