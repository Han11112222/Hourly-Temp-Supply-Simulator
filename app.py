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

st.title("🔥 DSE 공급량 예측 모델 _ 정밀기온 활용 ver")
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
# 3. 모델 학습 및 데이터 연산 파이프라인
# ==========================================
train_df = merged_df[merged_df['Year'].isin(train_years)].dropna(subset=['Daily_HDD', SHEET_TEMP_COL])
y_train = train_df[TARGET_COL]

# 모델 피팅
model_m1 = make_pipeline(LinearRegression())
model_m1.fit(train_df[['Daily_HDD']], y_train)

model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

# R2 및 계수 추출
coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_HDD']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

# --- 데이터셋 1: 과거 검증용 연산 ---
eval_df = merged_df[merged_df['Year'].isin(eval_years)].copy()
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[['Daily_HDD']])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)

# 차트용 월별 데이터
monthly_eval = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum', '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().rename(columns={TARGET_COL: '실제_공급량합계'})
monthly_eval = monthly_eval.set_index('Year_Month')

# ★ 수정된 부분: 파트 1 하단에 표출할 과거 검증용 '연도별' 합산 연산
yearly_eval = eval_df.groupby('Year').agg({
    TARGET_COL: 'sum', '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().rename(columns={TARGET_COL: '실제_공급량합계'})
yearly_eval['방법1_차이'] = yearly_eval['방법1_예측(정밀)'] - yearly_eval['실제_공급량합계']
yearly_eval['방법2_차이'] = yearly_eval['방법2_예측(단순)'] - yearly_eval['실제_공급량합계']
yearly_eval = yearly_eval[['Year', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법2_예측(단순)', '방법2_차이']].set_index('Year')

# --- 데이터셋 2: 미래 추정용 연산 ---
date_list = []
for y in future_years:
    dates = pd.date_range(start=f'{y}-01-01', end=f'{y}-12-31')
    temp_target_df = pd.DataFrame({'Date': dates, 'Year': dates.year, 'Month': dates.month, 'Day': dates.day})
    date_list.append(temp_target_df)
future_base_df = pd.concat(date_list, ignore_index=True)

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

# 연도별 합산 요약 데이터 생성 (미래 추정용)
yearly_future = future_df.groupby('Year').agg({
    '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().set_index('Year')


# ==========================================
# 4. 메인 대시보드 화면 구성 (수직 스크롤 구조)
# ==========================================

st.divider()

# ------------------------------------------
# 파트 1. 과거 모델 적합도 검증 (상단)
# ------------------------------------------
st.header("📊 [Part 1] 과거 모델 적합도 검증")
st.markdown(f"선택된 검증 연도({min(eval_years)}년~{max(eval_years)}년)의 실제 실적 데이터와 AI 모델들의 예측치를 정밀 대조합니다.")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown("### 🏆 [방법 1] 정밀 기온 (1차 선형 모델)")
    st.info(f"""
    **🎯 모델 학습 일치율 (R²): {train_r2_m1 * 100:.2f}%**
    
    **📉 도출된 1일 공급량 함수식:**
    $$y = {coef_m1[0]:.2f}x + {inter_m1:.0f}$$
    *(x = 1일 누적 난방도일(HDD))*
    
    💡 **해석 요약:** * 1시간 단위 온도를 기반으로 계산된 HDD 지표 활용
    * 기온과 공급량 사이의 비선형적(가속도 법칙) 특성을 이미 내포함
    * 복잡한 수식 없이 안정적인 1차 선형 모델만으로 높은 예측력 확보
    """)

with col_m2:
    st.markdown("### 📊 [방법 2] 단순 평균기온 (3차 다항식)")
    st.info(f"""
    **🎯 모델 학습 일치율 (R²): {train_r2_m2 * 100:.2f}%**
    
    **📉 도출된 1일 공급량 함수식:**
    $$y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$$
    *(x = 구글시트 일 평균기온)*
    
    💡 **해석 요약:** * 가공되지 않은 일 평균기온이 가진 설명력의 한계 보완
    * 동절기에 급증하는 공급량 특성에 맞추어 3차 곡선 함수 적용
    * 추위가 극심해질 때 수요가 기하급수적으로 늘어나는 민감도 포착
    """)

# 파트 1 그래프 높이 확대 (height=550 파라미터 유지)
chart_cols_eval = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_eval[chart_cols_eval], use_container_width=True, height=550)

# ★ 수정된 부분: 월별 표를 제거하고 '연도별' 적합도 요약 리포트 박스로 교체
st.subheader("📆 연도별 적합도 요약 리포트 (예측 차이 비교)")
st.dataframe(yearly_eval.style.format("{:,.0f}"), use_container_width=True)

csv_eval = yearly_eval.to_csv(index=True).encode('utf-8-sig')
st.download_button("📥 과거 적합도 검증 연도별 리포트 다운로드", data=csv_eval, file_name="과거적합도_연도별_요약리포트.csv", mime="text/csv")


st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------
# 파트 2. 미래 공급량 추정 시나리오 (하단)
# ------------------------------------------
st.header("🔮 [Part 2] 미래 공급량 추정 시나리오")
st.markdown(f"검증된 모델을 기반으로 아직 실적이 없는 미래 연도({min(future_years)}년~{max(future_years)}년)의 공급량을 시뮬레이션합니다.")

# 미래 시나리오 조건 하이라이트 박스
st.warning(f"""
💡 **미래 기온 추정 시나리오 설정 완료**
* **기온 산출 기준:** 과거 최근 **{y_years}개 연도** ({min(sim_base_years)}년 ~ {max(sim_base_years)}년)의 기후 패턴을 반영
* **방법 1 (정밀):** 최근 {y_years}년 동안의 동일 날짜, **시간대별 각각의 평균 기온**을 먼저 구한 후 미래 일자별 모의 HDD를 재계산하여 대입.
* **방법 2 (단순):** 최근 {y_years}년 동안의 동일 날짜 **일평균 기온들의 단순 평균값**을 미래 일자별 기온으로 대입.
""")

# 파트 2 그래프 높이 확대 (height=550 파라미터 유지)
chart_cols_future = ['방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_future[chart_cols_future], use_container_width=True, height=550)
st.subheader("🗂️ 월별 데이터 요약 리포트")
st.dataframe(monthly_future.style.format("{:,.0f}"), use_container_width=True)

# 파트 2 연도별 합산 요약표
st.subheader("📆 연도별 시나리오 합산 요약")
st.dataframe(yearly_future.style.format("{:,.0f}"), use_container_width=True)

csv_future = monthly_future.to_csv(index=True).encode('utf-8-sig')
st.download_button("📥 미래 시나리오 추정 리포트 다운로드", data=csv_future, file_name="미래시나리오_추정리포트.csv", mime="text/csv")
