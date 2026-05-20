import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
print("1. 데이터를 불러오는 중입니다...")
try:
    temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
except UnicodeDecodeError:
    temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
try:
    supply_df = pd.read_csv(sheet_url)
    print("   -> 구글 시트 데이터 로드 성공!")
except Exception as e:
    print(f"   -> 구글 시트 로드 실패: {e}")

# ==========================================
# ★ 이름표 무시하고 위치로 자동 추적 ★
DATE_COL_IN_SHEET = supply_df.columns[0] # 무조건 첫 번째 기둥
TARGET_COL = supply_df.columns[1]        # 무조건 두 번째 기둥
# ==========================================

# ---------------------------------------------------------
# 2. 기온 파생 변수 (HDD 등) 생성
# ---------------------------------------------------------
print("2. 1시간 단위 기온 데이터를 가공 중입니다...")
hour_cols = [f'Hour{i}' for i in range(1, 25)]

temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)

temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24
temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

# ---------------------------------------------------------
# 3. 데이터 병합 및 초강력 숫자 정제 (정규표현식)
# ---------------------------------------------------------
print("3. 기온과 공급량을 결합하고 숫자를 정제합니다...")
supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])

# 🚨 여기가 아까와 완전히 달라진 핵심 부분입니다! (정규표현식)
# 숫자(\d)와 소수점(.)을 제외한 모든 문자(콤마, 공백, 특수기호 등)를 싹 다 지웁니다.
supply_df[TARGET_COL] = supply_df[TARGET_COL].astype(str).str.replace(r'[^\d.]', '', regex=True)

# 지운 결과를 진짜 숫자(float)로 변환합니다. (에러 시 빈칸은 0으로 처리)
supply_df[TARGET_COL] = pd.to_numeric(supply_df[TARGET_COL], errors='coerce').fillna(0)

merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')
train_df = merged_df[(merged_df['Year'] >= 2015) & (merged_df['Year'] <= 2022)].copy()

features = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
X_train = train_df[features]
y_train = train_df[TARGET_COL]  

# ---------------------------------------------------------
# 4. 공급량 예측 모델 학습
# ---------------------------------------------------------
print("4. 비선형 AI 예측 모델 학습을 시작합니다...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   -> 모델 학습 완료!")

# ---------------------------------------------------------
# 5. 시뮬레이션: 23, 24, 25년 기온 적용
# ---------------------------------------------------------
print("5. 최근 3년 시나리오 시뮬레이션을 진행합니다...")
scenario_23 = temp_df[temp_df['Year'] == 2023].copy().reset_index(drop=True)
scenario_24 = temp_df[temp_df['Year'] == 2024].copy().reset_index(drop=True)
scenario_25 = temp_df[temp_df['Year'] == 2025].copy().reset_index(drop=True) 

min_len = min(len(scenario_23), len(scenario_24), len(scenario_25))

pred_23 = model.predict(scenario_23.loc[:min_len-1, features])
pred_24 = model.predict(scenario_24.loc[:min_len-1, features])
pred_25 = model.predict(scenario_25.loc[:min_len-1, features])

# ---------------------------------------------------------
# 6. 최종 예상 공급량 도출
# ---------------------------------------------------------
print("6. 결과를 통합하여 최종 데이터를 추출합니다...")
result_df = pd.DataFrame({
    'Month': scenario_23.loc[:min_len-1, 'Month'],
    'Day': scenario_23.loc[:min_len-1, 'Day'],
    'Pred_Supply_23': pred_23,
    'Pred_Supply_24': pred_24,
    'Pred_Supply_25': pred_25
})

result_df['Final_Expected_Supply'] = result_df[['Pred_Supply_23', 'Pred_Supply_24', 'Pred_Supply_25']].mean(axis=1)

print("\n=== ✨ 최종 시뮬레이션 추출 완료 (상위 5일치 미리보기) ===")
print(result_df.head(5))

result_df.to_csv('최종_시나리오_공급량예측.csv', index=False, encoding='utf-8-sig')
print("\n✅ '최종_시나리오_공급량예측.csv' 파일 저장 완료!")
