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

st.title("🔥 대성에너지 공급량 예측 모델 및 미래 시나리오 대시보드")
st.markdown("과거 기상 및 공급량 데이터를 기반으로 AI 모델의 **적합도(R²)**를 검증하고, 이를 바탕으로 **미래의 공급량 시나리오**를 추정합니다.")

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

    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
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
    "1. AI 학습 연도 선택",
    options=all_train_years,
    default=default_train_years
)

eval_years = st.sidebar.multiselect(
    "2. 과거 적합도 검증 연도 (실제 vs 예측)", 
    options=[2023, 2024, 2025], 
    default=[2024, 2025]
)

future_years = st.sidebar.multiselect(
    "3. 미래 시나리오 추정 연도 (2026~)", 
    options=[2026, 2027, 2028, 2029, 2030], 
    default=[2026, 2027, 2028]
)

max_data_year = int(merged_df['Year'].dropna().max())
y_years = st.sidebar.slider(
    "4. 미래 기온 추정 기준 (최근 Y년 평균)", 
    min_value=1, max_value=10, value=3, step=1
)
sim_base_years = list(range(max_data_year - y_years + 1, max_data_year + 1))

if not train_years or not eval_years or not future_years:
    st.warning("👈 좌측 패널에서 연도 설정을 완료해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 및 최상단 R2 지표 표시
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)].dropna(subset=['Daily_HDD', SHEET_TEMP_COL])
y_train = train_df[TARGET_COL]

# 모델 피팅
model_m1 = make_pipeline(LinearRegression())
model_m1.fit(train_df[['Daily_HDD']], y_train)

model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

# 수식 및 R2 스코어 추출
coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_HDD']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

st.divider()

# ★ 최상단 R2 KPI 대시보드
col_kpi1, col_kpi2 = st.columns(2)
col_kpi1.metric(label="🏆 [방법 1] 정밀 기온 (1차 선형) 학습 일치율 (R²)", value=f"{train_r2_m1 * 100:.2f} %")
col_kpi2.metric(label="📊 [방법 2] 단순 평균 (3차 다항) 학습 일치율 (R²)", value=f"{train_r2_m2 * 100:.2f} %")

st.divider()

# ==========================================
# 4. 데이터 셋업 (과거 검증용 / 미래 추정용 분리)
# ==========================================
# [데이터셋 1] 과거 검증용 (eval_years)
eval_df = merged_df[merged_df['Year'].isin(eval_years)].copy()
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[['Daily_HDD']])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)

monthly_eval = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum', '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().rename(columns={TARGET_COL: '실제_공급량합계'})
monthly_eval['방법1_차이'] = monthly_eval['방법1_예측(정밀)'] - monthly_eval['실제_공급량합계']
monthly_eval['방법2_차이'] = monthly_eval['방법2_예측(단순)'] - monthly_eval['실제_공급량합계']
monthly_eval = monthly_eval.set_index('Year_Month')

# [데이터셋 2] 미래 추정용 (future_years)
date_list = []
for y in future_years:
    dates = pd.date_range(start=f'{y}-01-01', end=f'{y}-12-31')
    temp_target_df = pd.DataFrame({'Date': dates, 'Year': dates.year, 'Month': dates.month, 'Day': dates.day})
    date_list.append(temp_target_df)
future_base_df = pd.concat(date_list, ignore_index=True)

# 미래 시나리오 기온 생성 (최근 Y년 기반)
scenario_temp_df = temp_df[temp_df['Year'].isin(sim_base_years)]
scenario_merged_df = merged_df[merged_df['Year'].isin(sim_base_years)]

hour_cols = [f'Hour{i}' for i in range(1, 25)]
sim_hourly_profile = scenario_temp_df.groupby(['Month', 'Day'])[hour_cols].mean().reset_index()
sim_hourly_profile['Daily_HDD_sim'] = sim_hourly_profile[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24

sim_daily_profile = scenario_merged_df.groupby(['Month', 'Day'])[SHEET_TEMP_COL].mean().reset_index()
sim_daily_profile.rename(columns={SHEET_TEMP_COL: f'{SHEET_TEMP_COL}_sim'}, inplace=True)

sim_profile = pd.merge(sim_hourly_profile[['Month', 'Day', 'Daily_HDD_sim']], sim_daily_profile, on=['Month', 'Day'])
future_df = pd.merge(future_base_df, sim_profile, on=['Month', 'Day'], how='left')

future_df['방법1_예측(정밀)'] = model_m1.predict(future_df[['Daily_HDD_sim']].rename(columns={'Daily_HDD_sim': 'Daily_HDD'}))
future_df['방법2_예측(단순)'] = model_m2.predict(future_df[[f'{SHEET_TEMP_COL}_sim']].rename(columns={f'{SHEET_TEMP_COL}_sim': SHEET_TEMP_COL}))
future_df['Year_Month'] = future_df['Date'].dt.to_period('M').astype(str)

monthly_future = future_df.groupby('Year_Month').agg({
    '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().set_index('Year_Month')

# ==========================================
# 5. 탭(Tab) 기반 화면 구성
# ==========================================
tab1, tab2 = st.tabs(["📊 [Part 1] 과거 모델 적합도 검증", "🔮 [Part 2] 미래 공급량 추정 시나리오"])

with tab1:
    st.subheader(f"🔍 {min(eval_years)}년 ~ {max(eval_years)}년 실제 실적 vs 모델 예측 비교")
    st.markdown(f"**[도출된 함수식]**\n* **방법 1 (정밀):** $y = {coef_m1[0]:.2f}x + {inter_m1:.0f}$\n* **방법 2 (단순):** $y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$")
    
    chart_cols_eval = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
    st.line_chart(monthly_eval[chart_cols_eval], use_container_width=True)
    
    st.dataframe(monthly_eval.style.format("{:,.0f}"), use_container_width=True)
    
    csv_eval = monthly_eval.to_csv(index=True).encode('utf-8-sig')
    st.download_button("📥 적합도 리포트 다운로드", data=csv_eval, file_name="과거적합도_검증리포트.csv", mime="text/csv")

with tab2:
    st.subheader(f"🚀 {min(future_years)}년 ~ {max(future_years)}년 시나리오 예측 트렌드")
    st.info(f"💡 **시나리오 기준:** 최근 {y_years}년({min(sim_base_years)}~{max(sim_base_years)})의 시간대별 평균 및 일평균 기온 패턴을 미래 달력에 매핑하여 산출했습니다.")
    
    chart_cols_future = ['방법1_예측(정밀)', '방법2_예측(단순)']
    st.line_chart(monthly_future[chart_cols_future], use_container_width=True)
    
    st.dataframe(monthly_future.style.format("{:,.0f}"), use_container_width=True)
    
    csv_future = monthly_future.to_csv(index=True).encode('utf-8-sig')
    st.download_button("📥 미래 시나리오 리포트 다운로드", data=csv_future, file_name="미래시나리오_추정리포트.csv", mime="text/csv")
