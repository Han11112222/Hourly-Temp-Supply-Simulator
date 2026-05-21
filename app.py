import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 스트림릿 웹페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="도시가스 공급량 시뮬레이터", layout="wide")

st.title("🔥 대성에너지 월별 공급량 시뮬레이션 및 모델 비교 대시보드")
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온(방법1)**과 **단순 평균기온(방법2)**의 월별 예측값을 실제 실적과 정밀 비교합니다.")

# ==========================================
# 1. 데이터 로드 및 전처리 (캐싱 적용)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    # 기온 데이터 로드
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

    # 구글 스프레드시트 공급량 데이터 로드
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    supply_df = pd.read_csv(sheet_url)

    DATE_COL = supply_df.columns[0]
    TARGET_COL = supply_df.columns[1]

    # 기온 파생 변수 (방법1용 HDD 포함)
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
    temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
    temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)
    temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    # 공급량 데이터 정제 (콤마 제거)
    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    # 데이터 병합
    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    return temp_df, merged_df, TARGET_COL

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    temp_df, merged_df, TARGET_COL = load_and_preprocess_data()

# ==========================================
# 2. 좌측 사이드바(Sidebar): 형님의 요구사항 반영 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

# 요구사항 1: 학습 기간 선택을 슬라이더에서 개별 연도 다중 선택 버튼으로 변경 (2021년 제외 용이)
all_train_years = sorted(merged_df['Year'].unique())
default_train_years = [y for y in all_train_years if y >= 2015 and y <= 2023 and y != 2021] # 2021년은 디폴트에서 자동 제외

train_years = st.sidebar.multiselect(
    "1. AI 학습 연도 선택 (특이 연도 제외 가능)",
    options=all_train_years,
    default=default_train_years
)

# 요구사항 2: 예측 시나리오 연도 디폴트값 10년(2016~2026년) 설정
all_sim_years = sorted(temp_df['Year'].unique())
default_sim_years = [y for y in range(2016, 2027) if y in all_sim_years]

sim_years = st.sidebar.multiselect(
    "2. 예측에 사용할 기온 시나리오 연도", 
    options=all_sim_years, 
    default=default_sim_years
)

if not train_years or not sim_years:
    st.warning("👈 좌측 패널에서 학습 연도와 기온 시나리오 연도를 각각 1개 이상 선택해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 (Daily 기준 학습)
# ==========================================
# 사용자가 선택한 연도 데이터만 필터링하여 학습
train_df = merged_df[merged_df['Year'].isin(train_years)]
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base (HDD 포함)
features_m1 = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
model_m1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_m1.fit(train_df[features_m1], y_train)

# [방법 2] 단순 일평균 기온 Base
features_m2 = ['Daily_Mean']
model_m2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_m2.fit(train_df[features_m2], y_train)

# ==========================================
# 4. 최근 3개년(2024~2026년) 타임라인 월별 시뮬레이션 및 실제값 검증
# ==========================================
# 최근 3개년의 기온 데이터를 기준으로 일별 예측값 도출
target_years = [2024, 2025, 2026]
eval_df = temp_df[temp_df['Year'].isin(target_years)].copy()

if len(eval_df) == 0:
    st.error("데이터에 2024~2026년 기온 정보가 포함되어 있지 않습니다.")
    st.stop()

eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[features_m1])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[features_m2])

# 실제 실적값 결합 (구글 시트에 실적이 존재하는 날짜 매칭)
actual_sub = merged_df[merged_df['Year'].isin(target_years)][['Date', TARGET_COL]]
eval_df = pd.merge(eval_df, actual_sub, on='Date', how='left')
eval_df[TARGET_COL] = eval_df[TARGET_COL].fillna(0) # 실적이 아직 없는 미래일 경우 0 처리

# 요구사항 3: 일별 뾰족한 라인 대신 '월별(Year-Month)'로 데이터 합산(Aggregation)
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)

# 월별 총 공급량으로 합산
monthly_df = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum',
    '방법1_예측(정밀)': 'sum',
    '방법2_예측(단순)': 'sum'
}).reset_index()

# 구글 시트에 실제 실적(0보다 큰 값)이 있는 구간만 골라 R2 스코어 정확도 계산
valid_actual = monthly_df[monthly_df[TARGET_COL] > 0]
if len(valid_actual) > 1:
    r2_m1 = r2_score(valid_actual[TARGET_COL], valid_actual['방법1_예측(정밀)'])
    r2_m2 = r2_score(valid_actual[TARGET_COL], valid_actual['방법2_예측(단순)'])
else:
    r2_m1, r2_m2 = 0, 0

# 그래프용 인덱스 설정
monthly_df = monthly_df.set_index('Year_Month')

# ==========================================
# 5. 대시보드 화면 렌더링 (UI)
# ==========================================
st.success("🎯 설정하신 조건에 맞춰 AI 월별 공급량 시뮬레이션이 성공적으로 완료되었습니다!")

# R2 지표 출력
st.markdown("### 🏆 월별 공급량 예측 정확도 비교 ($R^2$ Score)")
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="[방법 1] 정밀 기온 Base (HDD 포함)", value=f"{r2_m1:.4f}", delta="추천 방식")
with col_m2:
    st.metric(label="[방법 2] 단순 일평균 기온 Base", value=f"{r2_m2:.4f}")

st.divider()

# 시각화 차트 및 데이터 테이블 구성
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 2024년 ~ 2026년 월별 공급량 비교 트렌드 (실제값 vs 예측값)")
    
    # 실제 실적이 있는 데이터만 매칭하여 차트 컬럼 구성
    chart_cols = ['방법1_예측(정밀)', '방법2_예측(단순)']
    if monthly_df[TARGET_COL].sum() > 0:
        monthly_df['실제_공급량합계'] = monthly_df[TARGET_COL]
        chart_cols.insert(0, '실제_공급량합계')
        
    st.line_chart(monthly_df[chart_cols])

with col2:
    st.subheader("🗂️ 월별 데이터 요약 리포트")
    st.dataframe(monthly_df, height=450)

# 다운로드 기능
csv = monthly_df.to_csv(index=True).encode('utf-8-sig')
st.download_button(
    label="📥 월별 시뮬레이션 비교 리포트 다운로드",
    data=csv,
    file_name="대성에너지_월별_공급량_시뮬레이션_비교.csv",
    mime="text/csv"
)
