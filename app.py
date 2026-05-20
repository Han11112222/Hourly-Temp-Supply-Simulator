import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ==========================================
# [설정] 구글 시트의 컬럼명 세팅 (형님이 찾아주신 정답 반영!)
# ==========================================
TARGET_COL = '공급량합계'       # ★ 수정 완료: 모델이 예측할 정답(y) 컬럼명
DATE_COL_IN_SHEET = '날짜'      # 구글 시트의 날짜 컬럼명

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
print("1. 데이터를 불러오는 중입니다...")
try:
    # 한글 깨짐 방지를 위해 인코딩 처리
    temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 
except UnicodeDecodeError:
    temp_df = pd.read_csv('합산기온.csv', encoding='cp949') 

# 구글 스프레드시트 공급량 데이터 로드
sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
try:
    supply_df = pd.read_csv(sheet_url)
    print("   -> 구글 시트 데이터 로드 성공!")
except Exception as e:
    print(f"   -> 구글 시트 로드 실패. URL이나 권한을 확인해주세요: {e}")

# ---------------------------------------------------------
# 2. 기온 파생 변수 (HDD 등) 생성 (비선형성 타격량 반영)
# ---------------------------------------------------------
print("2. 1시간 단위 기온 데이터를 분석용 특성(Feature)으로 가공 중입니다...")
hour_cols = [f'Hour{i}' for i in range(1, 25)]

temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)

# ★ 핵심: 난방도일(HDD) 계산 (기준 온도 18도 기준)
temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24

# 병합을 위해 날짜를 Datetime 형식으로 통일
temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

# ---------------------------------------------------------
# 3. 데이터 병합 및 학습 세트 준비
# ---------------------------------------------------------
print("3. 기온 데이터와 구글 시트 공급량 데이터를 결합합니다...")
# 구글 시트의 날짜를 파이썬 날짜 형식으로 변환
supply_df['Date'] = pd.to_datetime(supply_df[DATE_COL_IN_SHEET])

# 기온 데이터와 공급량 데이터를 날짜(Date) 기준으로 병합 (교집합)
merged_df = pd.merge(temp_df, supply_df, on='Date', how='inner')

# 머신러닝 학습 기간 설정 (예: 2015년 ~ 2022년 데이터로 AI를 학습시킴)
train_df = merged_df[(merged_df['Year'] >= 2015) & (merged_df['Year'] <= 2022)].copy()

# 머신러닝 모델에 넣을 입력값(X)과 정답(y) 분리
features = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
X_train = train_df[features]
y_train = train_df[TARGET_COL]  # 여기서 아까 발생했던 KeyError 원천 차단!

# ---------------------------------------------------------
# 4. 공급량 예측 모델 학습 (Random Forest)
# ---------------------------------------------------------
print("4. 비선형 AI 예측 모델(Random Forest) 학습을 시작합니다...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("   -> 모델 학습 완료!")

# ---------------------------------------------------------
# 5. 시뮬레이션: 23, 24, 25년 기온을 각각 모델에 입력하여 예측
# ---------------------------------------------------------
print("5. 최근 3년(23~25년) 기온 시나리오 기반 시뮬레이션을 진행합니다...")
# 원본 기온 데이터에서 각 연도별 데이터 분리
scenario_23 = temp_df[temp_df['Year'] == 2023].copy().reset_index(drop=True)
scenario_24 = temp_df[temp_df['Year'] == 2024].copy().reset_index(drop=True)
scenario_25 = temp_df[temp_df['Year'] == 2025].copy().reset_index(drop=True) 

# 가장 데이터 개수가 적은 연도(주로 윤년 차이)에 길이를 맞춰줌
min_len = min(len(scenario_23), len(scenario_24), len(scenario_25))

# 각각의 기온 데이터로 개별 예측(A, B, C) 도출
pred_23 = model.predict(scenario_23.loc[:min_len-1, features])
pred_24 = model.predict(scenario_24.loc[:min_len-1, features])
pred_25 = model.predict(scenario_25.loc[:min_len-1, features])

# ---------------------------------------------------------
# 6. 최종 예상 공급량 도출 (A, B, C 예측값의 평균)
# ---------------------------------------------------------
print("6. 예측 결과를 통합하여 최종 데이터를 추출합니다...")
result_df = pd.DataFrame({
    'Month': scenario_23.loc[:min_len-1, 'Month'],
    'Day': scenario_23.loc[:min_len-1, 'Day'],
    'Pred_Supply_23': pred_23,
    'Pred_Supply_24': pred_24,
    'Pred_Supply_25': pred_25
})

# ★ 형님의 핵심 논리: "예측된 공급량의 3년 평균 산출 (평균의 함정 회피)"
result_df['Final_Expected_Supply'] = result_df[['Pred_Supply_23', 'Pred_Supply_24', 'Pred_Supply_25']].mean(axis=1)

print("\n=== ✨ 최종 시뮬레이션 추출 완료 (상위 5일치 미리보기) ===")
print(result_df.head(5))

# 결과를 CSV로 저장 (엑셀에서 열어보기 좋게 utf-8-sig 인코딩 사용)
file_name = '최종_시나리오_공급량예측.csv'
result_df.to_csv(file_name, index=False, encoding='utf-8-sig')
print(f"\n✅ '{file_name}' 파일이 폴더에 성공적으로 저장되었습니다. 수고하셨습니다 형님!")
