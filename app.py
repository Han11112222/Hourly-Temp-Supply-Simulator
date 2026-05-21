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

st.title("🔥 대성에너지 기온 기반 공급량 시뮬레이션 대시보드")
st.markdown("학습 기간과 시뮬레이션 기온 시나리오를 선택하여, **단순 평균기온 방식**과 **정밀 기온(가속도 법칙) 방식**의 예측 성능을 비교합니다.")

# ==========================================
# 1. 데이터 로드 및 전처리 (캐싱으로 속도 최적화)
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
# 2. 좌측 사이드바(Sidebar): 사용자 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

# 학습 기간 선택 슬라이더
all_years = sorted(merged_df['Year'].unique())
train_years = st.sidebar.slider(
    "1. AI 학습 기간 선택", 
    min_value=min(all_years), 
    max_value=max(all_years), 
    value=(2015, 2022)
)

# 시뮬레이션 연도 선택 (복수 선택 가능)
sim_years = st.sidebar.multiselect(
    "2. 예측에 사용할 기온 시나리오 연도 (평균 산출)", 
    options=sorted(temp_df['Year'].unique()), 
    default=[2023, 2024, 2025]
)

if not sim_years:
    st.warning("👈 좌측 패널에서 시나리오 연도를 1개 이상 선택해 주세요.")
    st.stop()

# ==========================================
# 3. 모델 학습 및 R2 성능 평가
# ==========================================
# 학습 데이터 필터링
train_df = merged_df[(merged_df['Year'] >= train_years[0]) & (merged_df['Year'] <= train_years[1])]
y_train = train_df[TARGET_COL]

# [방법 1] 정밀 기온 Base (비선형성 반영)
features_m1 = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
model_m1 = RandomForestRegressor(n_estimators=100, random_state=42)
model_m1.fit(train_df[features_m1], y_train)

# [방법 2] 단순 일평균 기온 Base
features_m2 = ['Daily_Mean']
model_m2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_m2.fit(train_df[features_m2], y_train)

# R2 평가를 위해 선택된 시뮬레이션 연도의 '실제 공급량' 데이터 추출 (실제 데이터가 있는 경우만)
test_df = merged_df[merged_df['Year'].isin(sim_years)]

if len(test_df) > 0:
    pred_test_m1 = model_m1.predict(test_df[features_m1])
    pred_test_m2 = model_m2.predict(test_df[features_m2])
    r2_m1 = r2_score(test_df[TARGET_COL], pred_test_m1)
    r2_m2 = r2_score(test_df[TARGET_COL], pred_test_m2)
else:
    r2_m1, r2_m2 = 0, 0

# ==========================================
# 4. 시뮬레이션 (선택한 연도들의 예측량 도출 및 평균)
# ==========================================
sim_list = []
for y in sim_years:
    df_y = temp_df[temp_df['Year'] == y].copy()
    df_y['방법1_예측량(정밀)'] = model_m1.predict(df_y[features_m1])
    df_y['방법2_예측량(단순)'] = model_m2.predict(df_y[features_m2])
    sim_list.append(df_y[['Month', 'Day', '방법1_예측량(정밀)', '방법2_예측량(단순)']])

# 선택한 모든 연도의 데이터를 합친 후, 월/일 기준으로 묶어서 평균 산출
concat_sim = pd.concat(sim_list)
final_sim = concat_sim.groupby(['Month', 'Day']).mean().reset_index()

# 윤년(2/29) 등 날짜 꼬임 방지를 위해 정렬 및 가상 날짜 인덱스 생성
final_sim = final_sim.sort_values(by=['Month', 'Day'])
final_sim['Date_Dummy'] = pd.to_datetime('2024-' + final_sim['Month'].astype(str) + '-' + final_sim['Day'].astype(str), errors='coerce')
final_sim = final_sim.dropna(subset=['Date_Dummy']).set_index('Date_Dummy')

# ==========================================
# 5. 스트림릿 화면 그리기 (UI)
# ==========================================
# R2 성능 비교 대시보드
st.markdown("### 🏆 예측 모델 성능 비교 ($R^2$ Score)")
st.info(f"선택하신 시나리오 기간({', '.join(map(str, sim_years))}년)의 실제 공급량 데이터와 각 모델의 예측값을 비교한 정확도입니다. (1에 가까울수록 완벽함)")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="[방법 1] 정밀 기온 Base (HDD 포함)", value=f"{r2_m1:.4f}", delta="추천 방식", delta_color="normal")
with col_m2:
    st.metric(label="[방법 2] 단순 일평균 기온 Base", value=f"{r2_m2:.4f}")

st.divider()

# 결과 차트 및 표
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 일별 시뮬레이션 공급량 트렌드 ({len(sim_years)}개년 평균)")
    # 두 방법의 예측 결과 그래프 비교
    st.line_chart(final_sim[['방법1_예측량(정밀)', '방법2_예측량(단순)']])

with col2:
    st.subheader("🗂️ 데이터 상세보기")
    display_df = final_sim[['Month', 'Day', '방법1_예측량(정밀)', '방법2_예측량(단순)']].reset_index(drop=True)
    st.dataframe(display_df, height=400)

# 엑셀 다운로드 버튼
csv = display_df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    label="📥 시뮬레이션 결과 엑셀(CSV) 다운로드",
    data=csv,
    file_name="비교_공급량시뮬레이션.csv",
    mime="text/csv"
)
