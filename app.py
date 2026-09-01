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

st.title("🔥 DSE 공급량 예측 모델 _ 정밀기온(HDD/CDD) 활용 ver")
st.markdown("과거 기상 및 공급량 데이터를 기반으로 AI 모델의 **적합도(R²)**를 검증하고, 이를 바탕으로 **미래의 공급량 시나리오**를 추정합니다.")

# ==========================================
# 공통: 방법2용 구글시트 일별 평균기온 → 월별 평균 집계
# ==========================================
@st.cache_data
def load_monthly_avg_temp():
    """
    구글시트(13HrIz6O...) : 일별 평균기온
    → 월별 평균기온으로 집계하여 방법2에 사용
    """
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    try:
        daily_temp_df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"❌ 기온 구글시트 로드 오류: {e}")
        st.stop()

    col_list = daily_temp_df.columns.tolist()
    date_cols = [c for c in col_list if '날짜' in c or 'date' in c.lower() or 'Date' in c]
    DATE_COL = date_cols[0] if date_cols else col_list[0]
    temp_cols = [c for c in col_list if '기온' in c or 'temp' in c.lower()]
    TEMP_COL = temp_cols[0] if temp_cols else col_list[1]

    daily_temp_df['Date'] = pd.to_datetime(daily_temp_df[DATE_COL])
    daily_temp_df['Year'] = daily_temp_df['Date'].dt.year
    daily_temp_df['Month'] = daily_temp_df['Date'].dt.month
    daily_temp_df[TEMP_COL] = pd.to_numeric(daily_temp_df[TEMP_COL], errors='coerce')

    monthly_temp = daily_temp_df.groupby(['Year', 'Month'])[TEMP_COL].mean().reset_index()
    monthly_temp.rename(columns={TEMP_COL: 'Monthly_Avg_Temp'}, inplace=True)
    return monthly_temp


# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
@st.cache_data
def load_and_preprocess_data(monthly_temp_df):
    """
    [수정] 공급량 소스: 1vS-a9X gid=0 (공급량 실적 탭) 합계컬럼
    방법1: CSV 시간별 기온 → HDD/CDD
    방법2: 구글시트 일별기온 → 월별평균 (monthly_temp_df)
    """
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8')
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949')

    # ★ 공급량 소스: 1vS-a9X gid=0 (공급량 실적 탭)
    sheet_url = "https://docs.google.com/spreadsheets/d/1vS-a9XrbjjIznHxntuFIM6hmml6qTlR2Cayw77p_Rao/export?format=csv&gid=0"
    try:
        supply_df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"❌ 공급량 구글시트 로드 오류: {e}")
        st.stop()

    col_list = supply_df.columns.tolist()
    DATE_COL_IN_SHEET = col_list[0]  # A열: 날짜

    # ★ 날짜/연/월/평균기온 4개 컬럼 제외한 나머지가 상품별 공급량
    # (컬럼명 기반으로 탐지하여 인덱스 오류 방지)
    skip_cols = set(col_list[:4])  # 날짜, 연, 월, 평균기온
    supply_cols = [c for c in col_list if c not in skip_cols]
    for c in supply_cols:
        supply_df[c] = pd.to_numeric(
            supply_df[c].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce'
        ).fillna(0)
    supply_df['전체합계'] = supply_df[supply_cols].sum(axis=1)
    TARGET_COL = '전체합계'

    # ★ 방법2용 SHEET_TEMP_COL은 monthly_temp_df의 컬럼명으로 고정
    SHEET_TEMP_COL = 'Monthly_Avg_Temp'

    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Daily_CDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(x - 26, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    supply_df['Date_parsed'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET], errors='coerce')
    supply_df = supply_df.dropna(subset=['Date_parsed'])
    supply_df['Year'] = supply_df['Date_parsed'].dt.year
    supply_df['Month'] = supply_df['Date_parsed'].dt.month

    # m+2 구조: 합계가 0인 행(미확정 실적) 제거
    supply_df = supply_df[supply_df[TARGET_COL] > 0]

    # Year/Month 기준 merge (컬럼 충돌 방지)
    supply_sub = supply_df[['Year', 'Month', TARGET_COL]].drop_duplicates(subset=['Year', 'Month'])
    merged_df = pd.merge(temp_df, supply_sub, on=['Year', 'Month'], how='inner')

    # 공급량 일평균 변환
    days_in_month = merged_df.groupby(['Year', 'Month'])[TARGET_COL].transform('count')
    merged_df[TARGET_COL] = merged_df[TARGET_COL] / days_in_month

    # 방법1 피처: HDD/CDD 월평균 평탄화
    merged_df['Daily_HDD'] = merged_df.groupby(['Year', 'Month'])['Daily_HDD'].transform('mean')
    merged_df['Daily_CDD'] = merged_df.groupby(['Year', 'Month'])['Daily_CDD'].transform('mean')

    # 방법2 피처: 구글시트 월별 평균기온 merge
    merged_df = pd.merge(merged_df, monthly_temp_df, on=['Year', 'Month'], how='left')

    return merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df


@st.cache_data
def load_and_preprocess_heating_data(monthly_temp_df):
    """
    [수정] 공급량 소스: 1vS-a9X gid=0 개별난방용 컬럼
    방법1: CSV 시간별 기온 → HDD/CDD
    방법2: 구글시트 일별기온 → 월별평균 (monthly_temp_df)
    원본 로직 유지 + 방법2 기온 소스만 구글시트 월별평균으로 교체
    """
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8')
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949')

    sheet_url_2 = "https://docs.google.com/spreadsheets/d/1vS-a9XrbjjIznHxntuFIM6hmml6qTlR2Cayw77p_Rao/export?format=csv&gid=0"
    try:
        supply_df = pd.read_csv(sheet_url_2)
    except Exception as e:
        st.error(f"❌ 구글 시트를 읽는 중 오류가 발생했습니다. 링크를 확인해 주세요. | 에러 내용: {e}")
        st.stop()

    col_list = supply_df.columns.tolist()

    year_col = '연' if '연' in col_list else ('Year' if 'Year' in col_list else col_list[1])
    month_col = '월' if '월' in col_list else ('Month' if 'Month' in col_list else col_list[2])

    # ★ 방법2용 SHEET_TEMP_COL은 monthly_temp_df의 컬럼명으로 고정
    SHEET_TEMP_COL = 'Monthly_Avg_Temp'
    TARGET_COL = '개별난방용'

    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Daily_CDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(x - 26, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    supply_df_renamed = supply_df.rename(columns={year_col: 'Year', month_col: 'Month'})
    supply_df_sub = supply_df_renamed[['Year', 'Month', TARGET_COL]]

    # m+2 구조: 공급량 0인 행(미확정 실적) 제거
    supply_df_sub = supply_df_sub[supply_df_sub[TARGET_COL] > 0]

    # 연/월 단위 결합 (원본 로직 유지)
    merged_df = pd.merge(temp_df, supply_df_sub, on=['Year', 'Month'], how='inner')

    # 💡 1. 타겟 변수(공급량)를 해당 월의 일수로 나누어 일평균 변환
    days_in_month = merged_df.groupby(['Year', 'Month'])[TARGET_COL].transform('count')
    merged_df[TARGET_COL] = merged_df[TARGET_COL] / days_in_month

    # 💡 2. 방법1 피처: HDD/CDD 월평균 평탄화 (원본 로직 유지)
    merged_df['Daily_HDD'] = merged_df.groupby(['Year', 'Month'])['Daily_HDD'].transform('mean')
    merged_df['Daily_CDD'] = merged_df.groupby(['Year', 'Month'])['Daily_CDD'].transform('mean')

    # 💡 3. [수정] 방법2 기온: CSV 덮어쓰기 제거 → 구글시트 월별 평균기온 사용
    merged_df = pd.merge(merged_df, monthly_temp_df, on=['Year', 'Month'], how='left')

    return merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df


# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

analysis_mode = st.sidebar.radio(
    "📊 분석 대상 선택",
    options=["1. 전체 공급량 분석", "2. 개별난방용 공급량 분석"],
    index=0
)

with st.spinner("데이터베이스를 불러오는 중입니다..."):
    monthly_temp_df = load_monthly_avg_temp()
    if analysis_mode == "1. 전체 공급량 분석":
        merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df = load_and_preprocess_data(monthly_temp_df)
    else:
        merged_df, TARGET_COL, SHEET_TEMP_COL, temp_df = load_and_preprocess_heating_data(monthly_temp_df)

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
train_df = merged_df[merged_df['Year'].isin(train_years)].dropna(subset=['Daily_HDD', 'Daily_CDD', SHEET_TEMP_COL])
y_train = train_df[TARGET_COL]

# 방법1: HDD/CDD 다중 선형 회귀
model_m1 = make_pipeline(LinearRegression())
model_m1.fit(train_df[['Daily_HDD', 'Daily_CDD']], y_train)

# 방법2: 월별 평균기온 3차 다항식
model_m2 = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
model_m2.fit(train_df[[SHEET_TEMP_COL]], y_train)

coef_m1 = model_m1.named_steps['linearregression'].coef_
inter_m1 = model_m1.named_steps['linearregression'].intercept_
train_r2_m1 = r2_score(y_train, model_m1.predict(train_df[['Daily_HDD', 'Daily_CDD']]))

coef_m2 = model_m2.named_steps['linearregression'].coef_
inter_m2 = model_m2.named_steps['linearregression'].intercept_
train_r2_m2 = r2_score(y_train, model_m2.predict(train_df[[SHEET_TEMP_COL]]))

# --- 데이터셋 1: 과거 검증용 연산 ---
eval_df = merged_df[merged_df['Year'].isin(eval_years)].copy()
eval_df['방법1_예측(정밀)'] = model_m1.predict(eval_df[['Daily_HDD', 'Daily_CDD']])
eval_df['방법2_예측(단순)'] = model_m2.predict(eval_df[[SHEET_TEMP_COL]])
eval_df['Year_Month'] = eval_df['Date'].dt.to_period('M').astype(str)

monthly_eval = eval_df.groupby('Year_Month').agg({
    TARGET_COL: 'sum', '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().rename(columns={TARGET_COL: '실제_공급량합계'})
monthly_eval['방법1_차이'] = monthly_eval['방법1_예측(정밀)'] - monthly_eval['실제_공급량합계']
monthly_eval['방법1_오차율(%)'] = (monthly_eval['방법1_차이'] / monthly_eval['실제_공급량합계']) * 100
monthly_eval['방법2_차이'] = monthly_eval['방법2_예측(단순)'] - monthly_eval['실제_공급량합계']
monthly_eval['방법2_오차율(%)'] = (monthly_eval['방법2_차이'] / monthly_eval['실제_공급량합계']) * 100

yearly_eval = eval_df.groupby('Year').agg({
    TARGET_COL: 'sum', '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index().rename(columns={TARGET_COL: '실제_공급량합계'})
yearly_eval['방법1_차이'] = yearly_eval['방법1_예측(정밀)'] - yearly_eval['실제_공급량합계']
yearly_eval['방법1_오차율(%)'] = (yearly_eval['방법1_차이'] / yearly_eval['실제_공급량합계']) * 100
yearly_eval['방법2_차이'] = yearly_eval['방법2_예측(단순)'] - yearly_eval['실제_공급량합계']
yearly_eval['방법2_오차율(%)'] = (yearly_eval['방법2_차이'] / yearly_eval['실제_공급량합계']) * 100

# --- 데이터셋 2: 미래 추정용 연산 ---
date_list = []
for y in future_years:
    dates = pd.date_range(start=f'{y}-01-01', end=f'{y}-12-31')
    temp_target_df = pd.DataFrame({'Date': dates, 'Year': dates.year, 'Month': dates.month, 'Day': dates.day})
    date_list.append(temp_target_df)
future_base_df = pd.concat(date_list, ignore_index=True)

scenario_temp_df = temp_df[temp_df['Year'].isin(sim_base_years)]

hour_cols = [f'Hour{i}' for i in range(1, 25)]
# 방법1 미래 기온: CSV 시간별 기온 → HDD/CDD 프로필
sim_hourly_profile = scenario_temp_df.groupby(['Month', 'Day'])[hour_cols].mean().reset_index()
sim_hourly_profile['Daily_HDD_sim'] = sim_hourly_profile[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
sim_hourly_profile['Daily_CDD_sim'] = sim_hourly_profile[hour_cols].apply(lambda x: np.maximum(x - 26, 0)).sum(axis=1) / 24

# 방법2 미래 기온: 구글시트 월별 평균기온의 최근 Y년 평균
sim_monthly_temp = monthly_temp_df[monthly_temp_df['Year'].isin(sim_base_years)]
sim_m2_temp = sim_monthly_temp.groupby('Month')['Monthly_Avg_Temp'].mean().reset_index()
sim_m2_temp.rename(columns={'Monthly_Avg_Temp': f'{SHEET_TEMP_COL}_sim'}, inplace=True)

sim_profile = sim_hourly_profile[['Month', 'Day', 'Daily_HDD_sim', 'Daily_CDD_sim']]
future_df = pd.merge(future_base_df, sim_profile, on=['Month', 'Day'], how='left')
future_df = pd.merge(future_df, sim_m2_temp, on='Month', how='left')

# 윤년(2월 29일) 결측치 방어
future_df[['Daily_HDD_sim', 'Daily_CDD_sim', f'{SHEET_TEMP_COL}_sim']] = \
    future_df[['Daily_HDD_sim', 'Daily_CDD_sim', f'{SHEET_TEMP_COL}_sim']].ffill()

future_df['방법1_예측(정밀)'] = model_m1.predict(
    future_df[['Daily_HDD_sim', 'Daily_CDD_sim']].rename(
        columns={'Daily_HDD_sim': 'Daily_HDD', 'Daily_CDD_sim': 'Daily_CDD'}))
future_df['방법2_예측(단순)'] = model_m2.predict(
    future_df[[f'{SHEET_TEMP_COL}_sim']].rename(
        columns={f'{SHEET_TEMP_COL}_sim': SHEET_TEMP_COL}))
future_df['Year_Month'] = future_df['Date'].dt.to_period('M').astype(str)

monthly_future_pred = future_df.groupby('Year_Month').agg({
    '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index()
yearly_future_pred = future_df.groupby('Year').agg({
    '방법1_예측(정밀)': 'sum', '방법2_예측(단순)': 'sum'
}).reset_index()

# 미래 연도의 실제 실적 데이터 가져오기
# ★ merged_df 대신 공급량 시트에서 직접 로드 (기온CSV 범위에 종속되지 않도록)
# ★ 분석 모드에 따라 전체합계 or 개별난방용 컬럼을 선택적으로 가져옴
@st.cache_data
def load_actual_supply(future_years_tuple, target_col):
    sheet_url = "https://docs.google.com/spreadsheets/d/1vS-a9XrbjjIznHxntuFIM6hmml6qTlR2Cayw77p_Rao/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
    except:
        return pd.DataFrame(columns=['Year_Month', 'Year', '실제_공급량합계'])

    col_list = df.columns.tolist()
    DATE_COL = col_list[0]  # A열: 날짜

    # 날짜 파싱으로 연/월 추출
    df['Date_parsed'] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=['Date_parsed'])
    df['Year'] = df['Date_parsed'].dt.year
    df['Month'] = df['Date_parsed'].dt.month
    df['Year_Month'] = df['Date_parsed'].dt.to_period('M').astype(str)

    if target_col == '개별난방용':
        # ★ 개별난방용 탭: 해당 컬럼 하나만 사용
        df['개별난방용'] = pd.to_numeric(
            df['개별난방용'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        ).fillna(0)
        df['실제_공급량합계'] = df['개별난방용']
    else:
        # ★ 전체 공급량 탭: E열(index=4)부터 상품별 합산
        supply_cols = col_list[4:]
        numeric_supply_cols = []
        for c in supply_cols:
            converted = pd.to_numeric(
                df[c].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
            if converted.notna().mean() > 0.5:
                df[c] = converted.fillna(0)
                numeric_supply_cols.append(c)
        df['실제_공급량합계'] = df[numeric_supply_cols].sum(axis=1)

    # m+2 구조: 실적이 없는 행(합계 0) 제거
    df = df[df['실제_공급량합계'] > 0]
    return df[df['Year'].isin(list(future_years_tuple))][
        ['Year', 'Month', 'Year_Month', '실제_공급량합계']
    ].copy()

future_actual_df = load_actual_supply(tuple(future_years), TARGET_COL)
if not future_actual_df.empty:
    actual_monthly = future_actual_df.groupby('Year_Month')['실제_공급량합계'].sum().reset_index()
    actual_yearly  = future_actual_df.groupby('Year')['실제_공급량합계'].sum().reset_index()
else:
    actual_monthly = pd.DataFrame(columns=['Year_Month', '실제_공급량합계'])
    actual_yearly  = pd.DataFrame(columns=['Year', '실제_공급량합계'])

monthly_future = pd.merge(monthly_future_pred, actual_monthly, on='Year_Month', how='left')
monthly_future['방법1_차이'] = monthly_future['방법1_예측(정밀)'] - monthly_future['실제_공급량합계']
monthly_future['방법1_오차율(%)'] = (monthly_future['방법1_차이'] / monthly_future['실제_공급량합계']) * 100
monthly_future['방법2_차이'] = monthly_future['방법2_예측(단순)'] - monthly_future['실제_공급량합계']
monthly_future['방법2_오차율(%)'] = (monthly_future['방법2_차이'] / monthly_future['실제_공급량합계']) * 100

yearly_future = pd.merge(yearly_future_pred, actual_yearly, on='Year', how='left')
yearly_future['방법1_차이'] = yearly_future['방법1_예측(정밀)'] - yearly_future['실제_공급량합계']
yearly_future['방법1_오차율(%)'] = (yearly_future['방법1_차이'] / yearly_future['실제_공급량합계']) * 100
yearly_future['방법2_차이'] = yearly_future['방법2_예측(단순)'] - yearly_future['실제_공급량합계']
yearly_future['방법2_오차율(%)'] = (yearly_future['방법2_차이'] / yearly_future['실제_공급량합계']) * 100


# ==========================================
# 공통 포맷 딕셔너리 & 소계 스타일 함수
# ==========================================
format_dict = {
    '실제_공급량합계': "{:,.0f}",
    '방법1_예측(정밀)': "{:,.0f}",
    '방법1_차이': "{:,.0f}",
    '방법1_오차율(%)': "{:.1f}%",
    '방법2_예측(단순)': "{:,.0f}",
    '방법2_차이': "{:,.0f}",
    '방법2_오차율(%)': "{:.1f}%"
}

def add_subtotal_style(df, numeric_cols, label_col, fmt):
    subtotal = {}
    for col in df.columns:
        if col == label_col:
            subtotal[col] = '📌 소계'
        elif col in numeric_cols:
            subtotal[col] = df[col].sum()
        else:
            subtotal[col] = ''
    # 오차율 소계 기준 재계산
    if '실제_공급량합계' in subtotal and subtotal['실제_공급량합계'] != 0:
        if '방법1_차이' in subtotal:
            subtotal['방법1_오차율(%)'] = (subtotal['방법1_차이'] / subtotal['실제_공급량합계']) * 100
        if '방법2_차이' in subtotal:
            subtotal['방법2_오차율(%)'] = (subtotal['방법2_차이'] / subtotal['실제_공급량합계']) * 100

    result_df = pd.concat([df, pd.DataFrame([subtotal])], ignore_index=True)

    def highlight(row):
        if row[label_col] == '📌 소계':
            return ['background-color: #1e3a5f; color: white; font-weight: bold;'] * len(row)
        return [''] * len(row)

    return result_df.style.apply(highlight, axis=1).format(
        {k: v for k, v in fmt.items() if k in df.columns}, na_rep='-'
    )

numeric_sum_cols = ['실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법2_예측(단순)', '방법2_차이']


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
    st.markdown("### 🏆 [방법 1] 정밀 기온 (다중 선형 모델)")
    st.info(f"""
    **🎯 모델 학습 일치율 (R²): {train_r2_m1 * 100:.2f}%**
    
    **📉 도출된 1일 공급량 함수식:**
    $y = {coef_m1[0]:.2f}x_1 + {coef_m1[1]:.2f}x_2 + {inter_m1:.0f}$
    *(x₁ = 난방도일(HDD), x₂ = 냉방도일(CDD))*
    
    👉 **"난방의 영향력 × 추운 정도 + 냉방의 영향력 × 더운 정도 + 날씨와 무관한 기본 소비량"**
    
    💡 **기술적 근거:**
    * **HDD(난방도일):** 기준 온도 18°C 적용
    * **CDD(냉방도일):** 기준 온도 26°C 적용
    * 1시간 단위 정밀 온도 데이터를 활용해 비선형적 특성을 선형 구간으로 완벽하게 해석함
    
    👉 **기온 소스: 합산기온.csv (1시간 단위 정밀 기온)**
    """)

with col_m2:
    st.markdown("### 📊 [방법 2] 단순 평균기온 (3차 다항식)")
    st.info(f"""
    **🎯 모델 학습 일치율 (R²): {train_r2_m2 * 100:.2f}%**
    
    **📉 도출된 1일 공급량 함수식:**
    $y = {coef_m2[2]:.2f}x^3 + {coef_m2[1]:.2f}x^2 + {coef_m2[0]:.2f}x + {inter_m2:.0f}$
    *(x = 구글시트 월별 평균기온)*
    
    💡 **해석 요약:**
    * 가공되지 않은 일 평균기온이 가진 설명력의 한계 보완
    * 동절기에 급증하는 공급량 특성에 맞추어 3차 곡선 함수 적용
    * 추위가 극심해질 때 수요가 기하급수적으로 늘어나는 민감도 포착
    
    👉 **기온 소스: 구글시트 일별 기온 → 월별 평균 집계**
    """)

chart_cols_eval = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
st.line_chart(monthly_eval.set_index('Year_Month')[chart_cols_eval], use_container_width=True, height=550)

st.subheader("🗂️ 월별 적합도 상세 리포트 (예측 차이 비교)")
display_eval_df = monthly_eval[['Year_Month', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법1_오차율(%)', '방법2_예측(단순)', '방법2_차이', '방법2_오차율(%)']]
st.dataframe(add_subtotal_style(display_eval_df, numeric_sum_cols, 'Year_Month', format_dict), use_container_width=True, hide_index=True)

st.subheader("📆 연도별 적합도 요약 리포트 (예측 차이 비교)")
display_yearly_eval = yearly_eval[['Year', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법1_오차율(%)', '방법2_예측(단순)', '방법2_차이', '방법2_오차율(%)']]
st.dataframe(add_subtotal_style(display_yearly_eval, numeric_sum_cols, 'Year', format_dict), use_container_width=True, hide_index=True)

csv_eval = display_eval_df.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 과거 적합도 검증 월별 리포트 다운로드", data=csv_eval, file_name="과거적합도_월별_검증리포트.csv", mime="text/csv")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------
# 파트 2. 미래 공급량 추정 시나리오 (하단)
# ------------------------------------------
st.header("🔮 [Part 2] 미래 공급량 추정 시나리오")
st.markdown(f"검증된 모델을 기반으로 아직 실적이 없는 미래 연도({min(future_years)}년~{max(future_years)}년)의 공급량을 시뮬레이션합니다.")

st.warning(f"""
💡 **미래 기온 추정 시나리오 설정 완료**
* **기온 산출 기준:** 과거 최근 **{y_years}개 연도** ({min(sim_base_years)}년 ~ {max(sim_base_years)}년)의 기후 패턴을 반영
* **방법 1 (정밀):** 최근 {y_years}년 동일 날짜, **시간대별 각각의 평균 기온**을 먼저 산출한 뒤, 이를 바탕으로 미래 일자별 모의 HDD(18°C) 및 CDD(26°C)를 계산하여 대입.
* **방법 2 (단순):** 최근 {y_years}년 동일 월의 **월별 평균기온 단순 평균값**을 미래 월별 기온으로 대입. (구글시트 기반)
""")

chart_cols_future = ['실제_공급량합계', '방법1_예측(정밀)', '방법2_예측(단순)']
available_chart_cols = [c for c in chart_cols_future if c in monthly_future.columns]
st.line_chart(monthly_future.set_index('Year_Month')[available_chart_cols], use_container_width=True, height=550)

st.subheader("🗂️ 월별 데이터 요약 리포트")
display_monthly_future = monthly_future[['Year_Month', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법1_오차율(%)', '방법2_예측(단순)', '방법2_차이', '방법2_오차율(%)']] \
    if '실제_공급량합계' in monthly_future.columns \
    else monthly_future[['Year_Month', '방법1_예측(정밀)', '방법2_예측(단순)']]
st.dataframe(add_subtotal_style(display_monthly_future, numeric_sum_cols, 'Year_Month', format_dict), use_container_width=True, hide_index=True)

st.subheader("📆 연도별 시나리오 합산 요약")
display_yearly_future = yearly_future[['Year', '실제_공급량합계', '방법1_예측(정밀)', '방법1_차이', '방법1_오차율(%)', '방법2_예측(단순)', '방법2_차이', '방법2_오차율(%)']] \
    if '실제_공급량합계' in yearly_future.columns \
    else yearly_future[['Year', '방법1_예측(정밀)', '방법2_예측(단순)']]
st.dataframe(add_subtotal_style(display_yearly_future, numeric_sum_cols, 'Year', format_dict), use_container_width=True, hide_index=True)

csv_future = display_monthly_future.to_csv(index=False).encode('utf-8-sig')
st.download_button("📥 미래 시나리오 추정 리포트 다운로드", data=csv_future, file_name="미래시나리오_추정리포트.csv", mime="text/csv")
