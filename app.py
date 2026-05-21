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
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온(방법1)**과 **단순 평균기온(방법2)**의 예측력을 실제 실적과 정밀 대조합니다.")

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    supply_df = pd.read_csv(sheet_url)

    col_list = supply_df.columns.tolist()
    DATE_COL_IN_SHEET = col_list[0]
    
    # 구글 시트의 기존 평균기온 컬럼 자동 인식 (방법2 용도)
    sheet_temp_cols = [c for c in col_list if '기온' in c]
    SHEET_TEMP_COL = sheet_temp_cols[0] if sheet_temp_cols else col_list[1]
    
    # 구글 시트의 실제 공급량 컬럼 자동 인식
    target_cols = [c for c in col_list if '공급량' in c or '합계' in c]
    TARGET_COL = target_cols[0] if target_cols else col_list[-1]

    # 합산기온.csv의 24시간 데이터를 '정밀 일 평균기온'으로 압축 (방법1 용도)
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    # 결측치 방어
    temp_df['Daily_Mean'] = temp_df['Daily_Mean'].ffill().bfill()

    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    # 데이터 병합 (기온 데이터와 실제 공급량 매칭)
    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    return merged_df, TARGET_COL, SHEET_TEMP_COL

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    merged_df, TARGET_COL, SHEET_TEMP_COL = load_and_preprocess_data()

# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

all_train_years = sorted(merged_df['Year'].unique())
default_train_years = [y for y in all_train_years if y >= 2015 and y <= 2023 and y != 2021]

train_years = st.sidebar.multiselect(
    "1. AI 학습 연도 선택 (특이 연도 제외 가능)",
    options=all_train_years,
    default=default_train_years
)

if not train_years:
    st.warning("👈 좌측 패널에서 학습 연도를 선택해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 (형님의 엑셀 수식과 100% 동일한 구조)
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)]
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base (24시간 평균 낸 Daily_Mean 단일 변수의 3차 다항식)
model_m1 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m1.fit(train_df[['Daily_Mean']], y_train)

# [방법 2] 기존 기온 Base (구글 시트 SHEET_TEMP_COL 단일 변수의 3차 다항식)
model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

# 모델의 수식(Coefficient) 및 학습 정확도(R2) 추출
coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_Mean']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

# ==========================================
# 4. 일별 1:1 예측 후 월별 합산 (최근 3개년 타임라인)
# ==========================================
target_years = [2024, 2025, 2026]
eval_df = merged_df[merged_df['Year'].isin(target_years)].copy()

if len(eval_df) == 0:
    st.error("데이터에 2024~2026년 실적 및 기온 정보가 매칭되지 않았습니다.")
    st.stop()

# 각 모델별 '1일 공급량' 예측
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[['Daily_Mean']])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])

# '1일 공급량'들을 월별로 합산(sum)
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)
monthly_df = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum',
    '방법1_예측(정밀)': 'sum',
    '방법2_예측(단순)': 'sum'
}).reset_index()

monthly_df.rename(columns={TARGET_COL: '실제_공급량합계'}, inplace=True)

valid_actual = monthly_df[monthly_df['실제_공급량합계'] > 0]
if len(valid_actual) > 1:
    r2_m1_monthly = r2_score(valid_actual['실제_공급량합계'], valid_actual['방법1_예측(정밀)'])
    r2_m2_monthly = r2_score(valid_actual['실제_공급량합계'], valid_actual['방법2_예측(단순)'])
else:
    r2_m1_monthly, r2_m2_monthly = 0, 0

monthly_df = monthly_df.set_index('Year_Month')

# ==========================================
# 5. 대시보드 화면 구성 (UI)
# ==========================================
st.success("🎯 모델 로직 전면 수정 완료! 엑셀과 동일한 3차 다항식으로 1일 공급량을 1:1 예측합니다.")

# 수식 공개 및 학습 R2 표시
col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### 🏆 [방법 1] 정밀 기온 (24시간 평균)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m1[2]:.2f}x^3 + {coef_m1[1]:.2f}x^2 + {coef_m1[0]:.2f}x + {inter_m1:.0f}$ \n\n *(학습 데이터 일치율 R²: {train_r2_m1:.4f})*")
with col_m2:
    st.markdown("### 📊 [방법 2] 단순 평균기온 (구글 시트)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$ \n\n *(학습 데이터 일치율 R²: {train_r2_m2:.4f})*")

st.divider()

# 차트
st.subheader("📈 2024년 ~ 2026년 월별 공급량 비교 트렌드 (실제 실적 스케일)")
chart_cols = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_df[chart_cols], use_container_width=True)

st.divider()

# 표
st.subheader("🗂️ 월별 데이터 요약 리포트")
st.dataframe(monthly_df, use_container_width=True)
