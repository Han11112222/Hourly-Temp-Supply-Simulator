import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 스트림릿 웹페이지 기본 설정
st.set_page_config(page_title="도시가스 공급량 시뮬레이터", layout="wide")

st.title("🔥 대성에너지 기온 기반 공급량 시뮬레이션 대시보드")
st.markdown("1시간 단위 세부 기온 데이터를 활용하여 **비선형성(추위의 가속도 법칙)**을 반영한 최근 3개년 시나리오별 공급량 예측 모델입니다.")

# ==========================================
# 데이터 로드 및 모델 학습 캐싱 (웹앱 속도 최적화)
# ==========================================
@st.cache_data
def load_and_train():
    # 1. 기온 데이터 로드
    try:
        temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
    except:
        temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

    # 2. 구글 스프레드시트 공급량 데이터 로드
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    supply_df = pd.read_csv(sheet_url)

    # 컬럼 위치 기반 자동 인식
    DATE_COL_IN_SHEET = supply_df.columns[0]
    TARGET_COL = supply_df.columns[1]

    # 3. 기온 파생 변수 및 HDD 생성
    hour_cols = [f'Hour{i}' for i in range(1, 25)]
    temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
    temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
    temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)
    temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
    temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

    # 4. 데이터 정제 및 병합
    supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])
    supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)
    supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

    merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
    
    # 학습 세트 분리 (2015~2022)
    train_df = merged_df[(merged_df['Year'] >= 2015) & (merged_df['Year'] <= 2022)].copy()
    features = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
    
    X_train = train_df[features]
    y_train = train_df[TARGET_COL]

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, temp_df, features

# 시동 걸기
with st.spinner("AI 모델 및 데이터를 불러오는 중입니다... 잠시만 기다려주세요."):
    model, temp_df, features = load_and_train()

# ---------------------------------------------------------
# 시뮬레이션 및 결과 생성
# ---------------------------------------------------------
scenario_23 = temp_df[temp_df['Year'] == 2023].copy().reset_index(drop=True)
scenario_24 = temp_df[temp_df['Year'] == 2024].copy().reset_index(drop=True)
scenario_25 = temp_df[temp_df['Year'] == 2025].copy().reset_index(drop=True) 

min_len = min(len(scenario_23), len(scenario_24), len(scenario_25))

pred_23 = model.predict(scenario_23.loc[:min_len-1, features])
pred_24 = model.predict(scenario_24.loc[:min_len-1, features])
pred_25 = model.predict(scenario_25.loc[:min_len-1, features])

result_df = pd.DataFrame({
    'Month': scenario_23.loc[:min_len-1, 'Month'],
    'Day': scenario_23.loc[:min_len-1, 'Day'],
    '23년 기온 기준 예측': pred_23,
    '24년 기온 기준 예측': pred_24,
    '25년 기온 기준 예측': pred_25
})
result_df['최종 예상 공급량(3년 평균)'] = result_df[['23년 기온 기준 예측', '24년 기온 기준 예측', '25년 기온 기준 예측']].mean(axis=1)

# ---------------------------------------------------------
# 스트림릿 화면 그리기 (UI 화면 구성)
# ---------------------------------------------------------
st.success("데이터 정제 및 AI 시뮬레이션이 성공적으로 완료되었습니다!")

# 레이아웃 분할 (좌측: 그래프, 우측: 데이터 표)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 일별 시뮬레이션 공급량 트렌드")
    # 대화형 라인 차트 그리기
    st.line_chart(result_df[['23년 기온 기준 예측', '24년 기온 기준 예측', '25년 기온 기준 예측', '최종 예상 공급량(3년 평균)']])

with col2:
    st.subheader("🗂️ 데이터 상세보기 (일별)")
    st.dataframe(result_df, height=400)

# 다운로드 버튼 제공
csv = result_df.to_csv(index=False).encode('utf-8-sig')
st.download_button(
    label="📥 시뮬레이션 결과 엑셀(CSV) 다운로드",
    data=csv,
    file_name="최종_시나리오_공급량예측.csv",
    mime="text/csv"
)
