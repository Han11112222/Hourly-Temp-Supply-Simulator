import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 스트림릿 웹페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="도시가스 공급량 시뮬레이터", layout="wide")

st.title("🔥 대성에너지 월별 공급량 시뮬레이션 및 모델 비교 대시보드")
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온(방법1)**과 **단순 평균기온(방법2)**의 예측값을 실제 실적과 비교합니다. (3차 다항식 회귀 모델 적용)")

# ==========================================
# 1. 데이터 로드 및 전처리 (캐싱 적용)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    supply_df = pd.read_csv(sheet_url)

    DATE_COL = supply_df.columns[0]
    TARGET_COL = supply_df.columns[1]

    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
    temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
    temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)
    temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    # ★ 에러 해결 핵심: 기온 데이터 중 빈칸(미래 날짜 등)이 있으면 앞/뒤 날짜 값으로 채워 넣음
    for col in ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']:
        temp_df[col] = temp_df[col].ffill().bfill()

    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    return temp_df, merged_df, TARGET_COL

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    temp_df, merged_df, TARGET_COL = load_and_preprocess_data()

# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

# 학습 연도 선택 (2021년 제외 디폴트)
all_train_years = sorted(merged_df['Year'].unique())
default_train_years = [y for y in all_train_years if y >= 2015 and y <= 2023 and y != 2021]

train_years = st.sidebar.multiselect(
    "1. 학습 연도 선택 (특이 연도 제외 가능)",
    options=all_train_years,
    default=default_train_years
)

# 시나리오 기온 10년 (2016~2026) 디폴트
all_sim_years = sorted(temp_df['Year'].unique())
default_sim_years = [y for y in range(2016, 2027) if y in all_sim_years]

sim_years = st.sidebar.multiselect(
    "2. 예측 시나리오 연도 (평균 산출용)", 
    options=all_sim_years, 
    default=default_sim_years
)

if not train_years or not sim_years:
    st.warning("👈 좌측 패널에서 학습 연도와 기온 시나리오 연도를 선택해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 (3차 다항식 회귀 모델)
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)]
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base (HDD 등 4개 변수의 3차 다항식)
features_m1 = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
model_m1 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m1.fit(train_df[features_m1], y_train)

# [방법 2] 단순 일평균 기온 Base (단일 변수의 3차 다항식)
features_m2 = ['Daily_Mean']
model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[features_m2], y_train)

# ==========================================
# 4. 시나리오 기반 타임라인 매핑 (24~26년)
# ==========================================
sim_df = temp_df[temp_df['Year'].isin(sim_years)].copy()
sim_df['Pred_M1'] = model_m1.predict(sim_df[features_m1])
sim_df['Pred_M2'] = model_m2.predict(sim_df[features_m2])

daily_profile = sim_df.groupby(['Month', 'Day'])[['Pred_M1', 'Pred_M2']].mean().reset_index()

target_years = [2024, 2025, 2026]
eval_df = temp_df[temp_df['Year'].isin(target_years)][['Year', 'Month', 'Day', 'Date']].copy()

eval_df = pd.merge(eval_df, daily_profile, on=['Month', 'Day'], how='left')

actual_sub = merged_df[merged_df['Year'].isin(target_years)][['Date', TARGET_COL]]
eval_df = pd.merge(eval_df, actual_sub, on='Date', how='left')
eval_df[TARGET_COL] = eval_df[TARGET_COL].fillna(0)

eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)
monthly_df = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum',
    'Pred_M1': 'sum',
    'Pred_M2': 'sum'
}).reset_index()

monthly_df.rename(columns={'Pred_M1': '방법1_예측(정밀)', 'Pred_M2': '방법2_예측(단순)'}, inplace=True)

valid_actual = monthly_df[monthly_df[TARGET_COL] > 0]
if len(valid_actual) > 1:
    r2_m1 = r2_score(valid_actual[TARGET_COL], valid_actual['방법1_예측(정밀)'])
    r2_m2 = r2_score(valid_actual[TARGET_COL], valid_actual['방법2_예측(단순)'])
else:
    r2_m1, r2_m2 = 0, 0

monthly_df = monthly_df.set_index('Year_Month')

# ==========================================
# 5. 대시보드 화면 구성 (UI)
# ==========================================
st.success("🎯 3차 다항식 회귀(Cubic Polynomial Regression) 기반 시뮬레이션이 완료되었습니다!")

# R2 지표
st.markdown("### 🏆 월별 공급량 예측 정확도 비교 ($R^2$ Score)")
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="[방법 1] 정밀 기온 Base (3차 다항식)", value=f"{r2_m1:.4f}", delta="설명 가능한 비선형 모델")
with col_m2:
    st.metric(label="[방법 2] 단순 일평균 기온 Base", value=f"{r2_m2:.4f}")

st.divider()

# 그래프
st.subheader("📊 2024년 ~ 2026년 월별 공급량 비교 트렌드")
chart_cols = ['방법1_예측(정밀)', '방법2_예측(단순)']
if monthly_df[TARGET_COL].sum() > 0:
    monthly_df['실제_공급량합계'] = monthly_df[TARGET_COL]
    chart_cols.insert(0, '실제_공급량합계')
    
st.line_chart(monthly_df[chart_cols], use_container_width=True)

st.divider()

# 데이터 표
st.subheader("🗂️ 월별 데이터 요약 리포트")
st.dataframe(monthly_df, use_container_width=True)

csv = monthly_df.to_csv(index=True).encode('utf-8-sig')
st.download_button(
    label="📥 시뮬레이션 결과 다운로드",
    data=csv,
    file_name="비교_공급량시뮬레이션_다항식.csv",
    mime="text/csv"
)
