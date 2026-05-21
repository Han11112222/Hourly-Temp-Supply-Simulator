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
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온(1차 선형)**과 **단순 평균기온(3차 다항식)**의 예측력을 실제 실적과 정밀 대조합니다.")

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
    
    # 구글 시트 기온 및 공급량 컬럼 자동 인식
    sheet_temp_cols = [c for c in col_list if '기온' in c]
    SHEET_TEMP_COL = sheet_temp_cols[0] if sheet_temp_cols else col_list[1]
    
    target_cols = [c for c in col_list if '공급량' in c or '합계' in c]
    TARGET_COL = target_cols[0] if target_cols else col_list[-1]

    # ★ 방법 1: 1시간 단위 데이터를 활용한 일일 누적 난방도일(HDD) 산출
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    # 데이터 병합
    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    return merged_df, TARGET_COL, SHEET_TEMP_COL

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    merged_df, TARGET_COL, SHEET_TEMP_COL = load_and_preprocess_data()

# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

all_train_years = sorted(merged_df['Year'].dropna().unique())
default_train_years = [y for y in all_train_years if y >= 2015 and y <= 2023 and y != 2021]

train_years = st.sidebar.multiselect(
    "1. AI 학습 연도 선택 (특이 연도 제외 가능)",
    options=all_train_years,
    default=default_train_years
)

all_sim_years = sorted(merged_df['Year'].dropna().unique())
default_sim_years = [y for y in range(2016, 2027) if y in all_sim_years]

sim_years = st.sidebar.multiselect(
    "2. 미래 예측용 시나리오 기준 연도", 
    options=all_sim_years, 
    default=default_sim_years
)

if not train_years or not sim_years:
    st.warning("👈 좌측 패널에서 연도 설정을 완료해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 (★ 완벽한 역할 분리 ★)
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)].dropna(subset=['Daily_HDD', SHEET_TEMP_COL])
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base: 이미 비선형성이 반영된 HDD는 1차 선형 회귀로 직행
model_m1 = make_pipeline(LinearRegression())
model_m1.fit(train_df[['Daily_HDD']], y_train)

# [방법 2] 기존 기온 Base: 날것의 평균기온은 3차 다항식 곡선 적용
model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

# 수식 및 R2 스코어 추출
coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_HDD']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

# ==========================================
# 4. 타임라인 매핑 및 시뮬레이션 예측
# ==========================================
target_years = [2024, 2025, 2026]
# ★ 수정된 부분: 'Daily_HDD'와 SHEET_TEMP_COL 컬럼을 복사할 때 포함시켰습니다.
eval_df = merged_df[merged_df['Year'].isin(target_years)][['Year', 'Month', 'Day', 'Date', TARGET_COL, 'Daily_HDD', SHEET_TEMP_COL]].copy()

# 시나리오 빈칸 채우기 로직 (2026년 등 데이터가 없는 구간 방어)
scenario_df = merged_df[merged_df['Year'].isin(sim_years)]
sim_profile = scenario_df.groupby(['Month', 'Day'])[['Daily_HDD', SHEET_TEMP_COL]].mean().reset_index()

eval_df = pd.merge(eval_df, sim_profile, on=['Month', 'Day'], how='left', suffixes=('', '_sim'))
eval_df['Daily_HDD'] = eval_df['Daily_HDD'].fillna(eval_df['Daily_HDD_sim'])
eval_df[SHEET_TEMP_COL] = eval_df[SHEET_TEMP_COL].fillna(eval_df[f'{SHEET_TEMP_COL}_sim'])

# 각 모델별 1일 예측
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[['Daily_HDD']])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])

# 월별 합산
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
st.success("🎯 모델 간섭 완벽 해결! 1시간 단위 정밀 가공 데이터에 안정적인 선형 모델을 매칭했습니다.")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### 🏆 [방법 1] 정밀 기온 (1차 선형 모델)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m1[0]:.2f}x + {inter_m1:.0f}$ \n\n *(x = 1일 누적 HDD, 학습 일치율 R²: {train_r2_m1:.4f})*\n\n 💡가속도 법칙이 이미 반영된 HDD 데이터를 사용하여 오버피팅을 방지했습니다.")
with col_m2:
    st.markdown("### 📊 [방법 2] 단순 평균기온 (3차 다항식)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$ \n\n *(x = 구글시트 일 평균기온, 학습 일치율 R²: {train_r2_m2:.4f})*")

st.divider()

st.subheader("📈 2024년 ~ 2026년 월별 공급량 비교 트렌드 (실제 실적 스케일)")
chart_cols = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_df[chart_cols], use_container_width=True)

st.divider()

st.subheader("🗂️ 월별 데이터 요약 리포트")
st.dataframe(monthly_df, use_container_width=True)

csv = monthly_df.to_csv(index=True).encode('utf-8-sig')
st.download_button(
    label="📥 시뮬레이션 결과 다운로드",
    data=csv,
    file_name="비교_공급량시뮬레이션_마스터.csv",
    mime="text/csv"
)
