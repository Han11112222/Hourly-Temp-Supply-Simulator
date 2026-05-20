import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ==========================================
# [설정] 구글 시트의 실제 공급량(타겟) 컬럼명을 아래에 적어주세요!
# 예: '공급량', '합계', 'Total' 등
# ==========================================
TARGET_COL = '공급량' # ★ 이 부분을 형님 시트에 맞게 꼭 수정해 주세요!
DATE_COL_IN_SHEET = '날짜' # ★ 구글 시트의 날짜 컬럼명 (예: '일자', 'Date' 등)

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
print("데이터를 불러오는 중입니다...")
# 기온 데이터 로드 
try:
    temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
except:
    temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

# 구글 스프레드시트 공급량 데이터 로드
sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
try:
    supply_df = pd.read_csv(sheet_url)
    print("구글 시트 로드 성공!")
except Exception as e:
    print(f"구글 시트 로드 실패. URL이나 권한을 확인해주세요: {e}")

# ---------------------------------------------------------
# 2. 기온 파생 변수 (HDD 등) 생성 (KOGAS 비선형 로직 반영)
# ---------------------------------------------------------
hour_cols = [f'Hour{i}' for i in range(1, 25)]

temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)

# ★ 핵심: 난방도일(HDD) 계산 (기준 온도 18도)
temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24

# 병합을 위해 날짜를 Datetime 형식으로 통일
temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

# ---------------------------------------------------------
# 3. 데이터 병합 및 학습 세트 준비 (에러 수정 완료)
# ---------------------------------------------------------
# 구글 시트의 날짜 컬럼을 Datetime으로 변환 (에러 방지)
supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])

# 기온 데이터와 공급량 데이터 병합 (날짜 기준)
merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')

# 학습 데이터 추출 (예: 2015년 ~ 2022년 데이터로 학습)
train_df = merged_df[(merged_df['Year'] >= 2015) & (merged_df['Year'] <= 2022)].copy()

# 머신러닝에 넣을 특징(X)과 정답(y) 분리
features = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
X_train = train_df[features]

# ★ 에러 수정 부분: 날짜가 아닌 정확한 '숫자' 컬럼이 y에 들어가도록 지정
y_train = train_df[TARGET_COL] 

# ---------------------------------------------------------
# 4. 공급량 예측 모델 학습 (Random Forest)
# ---------------------------------------------------------
print("AI 모델 학습을 시작합니다...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("모델 학습 완료!")

# ---------------------------------------------------------
# 5. 형님의 시나리오 시뮬레이션 (23, 24, 25년 기온 적용)
# ---------------------------------------------------------
# 원본 기온 데이터(temp_df)에서 각 연도별 데이터 분리
scenario_23 = temp_df[temp_df['Year'] == 2023].copy().reset_index(drop=True)
scenario_24 = temp_df[temp_df['Year'] == 2024].copy().reset_index(drop=True)
scenario_25 = temp_df[temp_df['Year'] == 2025].copy().reset_index(drop=True) 

# 각각의 기온 시나리오로 공급량 예측
# (데이터가 존재하는 길이까지만 잘라서 예측하여 윤년/길이 불일치 에러 방지)
min_len = min(len(scenario_23), len(scenario_24), len(scenario_25))

pred_23 = model.predict(scenario_23.loc[:min_len-1, features])
pred_24 = model.predict(scenario_24.loc[:min_len-1, features])
pred_25 = model.predict(scenario_25.loc[:min_len-1, features])

# ---------------------------------------------------------
# 6. 최종 3년 평균 공급량 도출 (가속도 법칙 오류 방지 로직)
# ---------------------------------------------------------
result_df = pd.DataFrame({
    'Month': scenario_23.loc[:min_len-1, 'Month'],
    'Day': scenario_23.loc[:min_len-1, 'Day'],
    'Pred_Supply_23': pred_23,
    'Pred_Supply_24': pred_24,
    'Pred_Supply_25': pred_25
})

# 3개의 예측 공급량을 평균 냄 (이것이 최종 타겟 데이터!)
result_df['Final_Expected_Supply'] = result_df[['Pred_Supply_23', 'Pred_Supply_24', 'Pred_Supply_25']].mean(axis=1)

print("\n=== 최종 공급량 시뮬레이션 결과 (상위 10일) ===")
print(result_df.head(10))

# 결과를 CSV로 저장하여 회의 자료용 엑셀로 활용
result_df.to_csv('최종_공급량_시뮬레이션결과.csv', index=False, encoding='utf-8-sig')
print("\n'최종_공급량_시뮬레이션결과.csv' 파일이 성공적으로 저장되었습니다.")
