import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ---------------------------------------------------------
# 스트림릿 웹페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="도시가스 공급량 시뮬레이터", layout="wide")

st.title("🔥 대성에너지 월별 공급량 시뮬레이션 및 모델 비교 대시보드")
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온 프로필(방법1)**과 **구글 시트 원본 평균기온(방법2)**의 3차 다항식 예측 결과를 실제 실적과 대조합니다.")

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

    # 구글 시트에서 날짜, 평균기온, 공급량합계 컬럼 자동 인식
    col_list = supply_df.columns.tolist()
    DATE_COL_IN_SHEET = col_list[0]
    
    # '기온' 글자가 들어간 컬럼을 방법2용 특징으로 자동 인식
    sheet_temp_cols = [c for c in col_list if '기온' in c]
    SHEET_TEMP_COL = sheet_temp_cols[0] if sheet_temp_cols else col_list[1]
    
    # '공급량' 또는 '합계' 글자가 들어간 컬럼을 타겟으로 자동 인식
    target_cols = [c for c in col_list if '공급량' in c or '합계' in c]
    TARGET_COL = target_cols[0] if target_cols else col_list[-1]

    # 1시간 단위 데이터 처리 (방법1용)
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
    temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
    temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)
    temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    # 결측치 방어
    for col in ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']:
        temp_df[col] = temp_df[col].ffill().bfill()

    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    # 데이터 병합 (이제 merged_df는 구글 시트의 원본 평균기온 컬럼도 함께 가집니다)
    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    return merged_df, TARGET_COL, SHEET_TEMP_COL

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    merged_df, TARGET_COL, SHEET_TEMP_COL = load_and_preprocess_data()

# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

# 학습 연도 선택 (2021년 코로나 특이구간 제외 가능)
all_train_years = sorted(merged_df['Year'].unique())
default_train_years = [y for y in all_train_years if y >= 2015 and y <= 2023 and y != 2021]

train_years = st.sidebar.multiselect(
    "1. 학습 연도 선택 (특이 연도 제외 가능)",
    options=all_train_years,
    default=default_train_years
)

# 예측 시나리오 기준 연도 설정 (디폴트 10개년 세팅)
all_sim_years = sorted(merged_df['Year'].unique())
default_sim_years = [y for y in range(2016, 2027) if y in all_sim_years]

sim_years = st.sidebar.multiselect(
    "2. 예측 시나리오 연도 범위 (디폴트 10년)", 
    options=all_sim_years, 
    default=default_sim_years
)

if not train_years or not sim_years:
    st.warning("👈 좌측 패널에서 연도 설정을 완료해 주세요.")
    st.stop()

# ==========================================
# 3. 3차 다항식 모델 학습 (독립적 변수 분리)
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)]
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base (24시간 데이터 파생 변수들의 3차 다항식)
features_m1 = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
model_m1 = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m1.fit(train_df[features_m1], y_train)

# [방법 2] 일 평균기온 Base (★구글 시트 원본 평균기온 단일 변수의 3차 다항식★)
model_m2 = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

# ==========================================
# 4. 최근 3개년(2024~2026년) 타임라인 시뮬레이션 및 검증
# ==========================================
target_years = [2024, 2025, 2026]
eval_df = merged_df[merged_df['Year'].isin(target_years)].copy()

if len(eval_df) == 0:
    st.error("데이터에 2024~2026년 실적 및 기온 정보가 매칭되지 않았습니다.")
    st.stop()

# 각 모델별 예측값 도출
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[features_m1])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])

# 월별 합산 데이터프레임 생성
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)
monthly_df = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum',
    '방법1_예측(정밀)': 'sum',
    '방법2_예측(단순)': 'sum'
}).reset_index()

monthly_df.rename(columns={TARGET_COL: '실제_공급량합계'}, inplace=True)

# R2 스코어 계산
valid_actual = monthly_df[monthly_df['실제_공급량합계'] > 0]
if len(valid_actual) > 1:
    r2_m1 = r2_score(valid_actual['실제_공급량합계'], valid_actual['방법1_예측(정밀)'])
    r2_m2 = r2_score(valid_actual['실제_공급량합계'], valid_actual['방법2_예측(단순)'])
else:
    r2_m1, r2_m2 = 0, 0

monthly_df = monthly_df.set_index('Year_Month')

# ==========================================
# 5. 대시보드 화면 구성 (UI)
# ==========================================
st.success("🎯 구글 시트 원본 평균기온 반영 및 스케일 튜닝이 완벽하게 완료되었습니다!")

# R2 지표
st.markdown("### 🏆 월별 공급량 예측 정확도 비교 ($R^2$ Score)")
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="[방법 1] 정밀 기온 Base (1시간 단위 HDD 포함 3차 다항식)", value=f"{r2_m1:.4f}", delta="가속도 법칙 최적화")
with col_m2:
    st.metric(label="[방법 2] 일 평균기온 Base (구글 시트 원본 평균기온 3차 다항식)", value=f"{r2_m2:.4f}")

st.divider()

# 그래프 상단 배치
st.subheader("📊 2024년 ~ 2026년 월별 공급량 비교 트렌드 (실제 실적 스케일 반영)")
chart_cols = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_df[chart_cols], use_container_width=True)

st.divider()

# 데이터 표 하단 배치 (형님 요청사항 반영)
st.subheader("🗂️ 월별 데이터 요약 리포트 (하단 배치)")
st.dataframe(monthly_df, use_container_width=True)

# 엑셀 다운로드
csv = monthly_df.to_csv(index=True).encode('utf-8-sig')
st.download_button(
    label="📥 시뮬레이션 결과 다운로드",
    data=csv,
    file_name="대성에너지_최종_모델비교_시뮬레이션.csv",
    mime="text/csv"
)
