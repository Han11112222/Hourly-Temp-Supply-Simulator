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
st.markdown("특이 연도를 제외한 맞춤형 AI 학습을 진행하고, **정밀 기온(1차 선형)**과 **단순 평균기온(3차 다항식)**의 예측력을 미래 시나리오에 적용합니다.")

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

    # 1시간 단위 데이터를 활용한 일일 누적 난방도일(HDD) 산출
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    # 데이터 병합
    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    # ★ 수정: 방법1의 '시간대별 평균'을 구하기 위해 temp_df 도 반환
    return merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df = load_and_preprocess_data()

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

# ★ 추가/수정: 미래 기온 추정용 최근 Y년 선택 슬라이더
max_data_year = int(merged_df['Year'].dropna().max())
y_years = st.sidebar.slider(
    "2. 미래 기온 추정 기준 (최근 Y년 평균)", 
    min_value=1, max_value=10, value=3, step=1,
    help="선택한 연수만큼의 과거 데이터를 평균 내어 미래 시나리오 기온으로 활용합니다."
)
sim_base_years = list(range(max_data_year - y_years + 1, max_data_year + 1))
st.sidebar.info(f"💡 설정됨: {min(sim_base_years)}년 ~ {max(sim_base_years)}년의 평균 패턴 적용")

target_years = st.sidebar.multiselect(
    "3. 그래프 및 리포트 출력 대상 연도", 
    options=[2024, 2025, 2026, 2027, 2028], 
    default=[2026, 2027, 2028]
)

if not train_years or not target_years:
    st.warning("👈 좌측 패널에서 연도 설정을 완료해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)].dropna(subset=['Daily_HDD', SHEET_TEMP_COL])
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base
model_m1 = make_pipeline(LinearRegression())
model_m1.fit(train_df[['Daily_HDD']], y_train)

# [방법 2] 기존 기온 Base
model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_HDD']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

# ==========================================
# 4. 타임라인 매핑 및 시나리오 프로필 생성
# ==========================================
# ★ 수정: 선택한 미래 연도(2026~)에 대한 빈 달력 뼈대 생성
date_list = []
for y in target_years:
    dates = pd.date_range(start=f'{y}-01-01', end=f'{y}-12-31')
    temp_target_df = pd.DataFrame({'Date': dates})
    temp_target_df['Year'] = temp_target_df['Date'].dt.year
    temp_target_df['Month'] = temp_target_df['Date'].dt.month
    temp_target_df['Day'] = temp_target_df['Date'].dt.day
    date_list.append(temp_target_df)

eval_base_df = pd.concat(date_list, ignore_index=True)

# 기존 실제 데이터 매핑 (미래 연도는 자연스럽게 NaN 처리됨)
eval_df = pd.merge(eval_base_df, merged_df[['Date', TARGET_COL, 'Daily_HDD', SHEET_TEMP_COL]], on='Date', how='left')

# --- 시나리오 프로필 구축 (최근 Y년 기반) ---
scenario_temp_df = temp_df[temp_df['Year'].isin(sim_base_years)]
scenario_merged_df = merged_df[merged_df['Year'].isin(sim_base_years)]

# 방법 1 시나리오: 시간대별 각각 평균 후 HDD 도출
hour_cols = [f'Hour{i}' for i in range(1, 25)]
sim_hourly_profile = scenario_temp_df.groupby(['Month', 'Day'])[hour_cols].mean().reset_index()
sim_hourly_profile['Daily_HDD_sim'] = sim_hourly_profile[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24

# 방법 2 시나리오: 일평균 기온의 단순 평균
sim_daily_profile = scenario_merged_df.groupby(['Month', 'Day'])[SHEET_TEMP_COL].mean().reset_index()
sim_daily_profile.rename(columns={SHEET_TEMP_COL: f'{SHEET_TEMP_COL}_sim'}, inplace=True)

# 시나리오 매핑
sim_profile = pd.merge(sim_hourly_profile[['Month', 'Day', 'Daily_HDD_sim']], sim_daily_profile, on=['Month', 'Day'])
eval_df = pd.merge(eval_df, sim_profile, on=['Month', 'Day'], how='left')

# 기온 인풋 데이터 결정 (실제 데이터가 없으면 시나리오 값 투입)
eval_df['Daily_HDD_Input'] = eval_df['Daily_HDD'].fillna(eval_df['Daily_HDD_sim'])
eval_df['TEMP_Input'] = eval_df[SHEET_TEMP_COL].fillna(eval_df[f'{SHEET_TEMP_COL}_sim'])

# 예측 수행 (컬럼명 불일치 에러 방지를 위해 rename 처리)
pred_m1_df = eval_df[['Daily_HDD_Input']].rename(columns={'Daily_HDD_Input': 'Daily_HDD'})
pred_m2_df = eval_df[['TEMP_Input']].rename(columns={'TEMP_Input': SHEET_TEMP_COL})

eval_df['방법1_예측(정밀)'] = model_m1.predict(pred_m1_df)
eval_df['방법2_예측(단순)'] = model_m2.predict(pred_m2_df)

# ==========================================
# 5. 리포트 집계 및 대시보드 UI
# ==========================================
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)
monthly_df = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum',
    '방법1_예측(정밀)': 'sum',
    '방법2_예측(단순)': 'sum'
}).reset_index()

monthly_df.rename(columns={TARGET_COL: '실제_공급량합계'}, inplace=True)

# ★ 미래 연도 빈 데이터가 0으로 합산되어 그래프 스케일을 망치는 것을 방지
monthly_df['실제_공급량합계'] = monthly_df['실제_공급량합계'].replace(0, np.nan)

monthly_df['방법1_차이'] = monthly_df['방법1_예측(정밀)'] - monthly_df['실제_공급량합계']
monthly_df['방법2_차이'] = monthly_df['방법2_예측(단순)'] - monthly_df['실제_공급량합계']

monthly_df = monthly_df[['Year_Month', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법2_예측(단순)', '방법2_차이']]
monthly_df = monthly_df.set_index('Year_Month')

# --- UI 화면 ---
st.success("🎯 모델 간섭 완벽 해결! 시간대별 정밀 합산을 통한 미래 HDD 시나리오를 적용했습니다.")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### 🏆 [방법 1] 정밀 기온 (1차 선형 모델)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m1[0]:.2f}x + {inter_m1:.0f}$ \n\n *(x = 1일 누적 HDD, 학습 일치율 R²: {train_r2_m1:.4f})*")
with col_m2:
    st.markdown("### 📊 [방법 2] 단순 평균기온 (3차 다항식)")
    st.info(f"**도출된 1일 공급량 함수식:**\n\n $y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$ \n\n *(x = 구글시트 일 평균기온, 학습 일치율 R²: {train_r2_m2:.4f})*")

st.divider()

st.subheader(f"📈 {min(target_years)}년 ~ {max(target_years)}년 월별 공급량 시뮬레이션 트렌드")
chart_cols = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_df[chart_cols], use_container_width=True)

st.divider()

st.subheader("🗂️ 월별 데이터 요약 리포트")
st.dataframe(monthly_df.style.format("{:,.0f}", na_rep="-"), use_container_width=True)

csv = monthly_df.to_csv(index=True).encode('utf-8-sig')
st.download_button(
    label="📥 시나리오 시뮬레이션 결과 다운로드",
    data=csv,
    file_name=f"공급량시나리오_예측_{min(target_years)}_{max(target_years)}.csv",
    mime="text/csv"
)
