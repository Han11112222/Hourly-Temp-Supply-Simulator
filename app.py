import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------
# 기온 데이터 로드 (형님이 업로드하신 파일)
# (주의: 한글 인코딩 문제 발생 시 'utf-8' 대신 'cp949' 또는 'euc-kr' 사용)
temp_df = pd.read_csv('합산기온.csv', encoding='utf-8') 

# 구글 스프레드시트 공급량 데이터 로드 (공유된 URL을 csv export용으로 변환)
sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
supply_df = pd.read_csv(sheet_url)

# ---------------------------------------------------------
# 2. 기온 데이터 전처리 및 파생 변수 생성 (Feature Engineering)
# ---------------------------------------------------------
# Hour1 ~ Hour24 컬럼명 리스트화
hour_cols = [f'Hour{i}' for i in range(1, 25)]

# 시간대별 데이터를 바탕으로 일평균, 일최고, 일최저 기온 추출
temp_df['Daily_Mean'] = temp_df[hour_cols].mean(axis=1)
temp_df['Daily_Max']  = temp_df[hour_cols].max(axis=1)
temp_df['Daily_Min']  = temp_df[hour_cols].min(axis=1)

# ★ 핵심: 난방도일(HDD) 계산 (기준 온도 18도)
# 24시간 각각 18도보다 낮은 만큼의 차이를 구해서 하루치 누적(평균)
temp_df['HDD'] = temp_df[hour_cols].apply(lambda x: np.maximum(18 - x, 0)).sum(axis=1) / 24

# 날짜 컬럼 생성 (병합을 위함)
temp_df['Date'] = pd.to_datetime(temp_df[['Year', 'Month', 'Day']])

# ---------------------------------------------------------
# 3. 공급량 데이터와 병합 및 학습 기간 설정
# ---------------------------------------------------------
# ※ 구글 시트의 날짜 컬럼명에 맞게 'Date'로 변경이 필요합니다.
# supply_df['Date'] = pd.to_datetime(supply_df['구글시트_날짜컬럼명'])
# merged_df = pd.merge(supply_df, temp_df, on='Date', how='inner')

# (아래는 병합이 완료된 merged_df가 있다고 가정한 코드입니다. 실사용 시 merged_df로 변수명을 맞춰주세요)
# 편의상 temp_df를 그대로 사용하여 시뮬레이션하겠습니다.
train_df = temp_df[(temp_df['Year'] >= 2015) & (temp_df['Year'] <= 2022)].copy() 
# supply_df가 병합되었다면 타겟 변수는 supply_df의 ['공급량컬럼명']이 됩니다.
# 여기서는 코드가 바로 돌아가도록 가상의 공급량(Target)을 만들겠습니다.
y_train = 100 + train_df['HDD'] * 15 + np.random.normal(0, 20, len(train_df)) 

features = ['Daily_Mean', 'Daily_Max', 'Daily_Min', 'HDD']
X_train = train_df[features]

# ---------------------------------------------------------
# 4. 공급량 예측 모델 학습 (비선형성 반영에 강한 Random Forest 사용)
# ---------------------------------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 5. 시뮬레이션: 최근 3년(23, 24, 25년) 기온 시나리오 개별 예측
# ---------------------------------------------------------
# 각 연도의 데이터를 분리 (윤년 등 날짜 개수가 다를 수 있으므로 월/일 기준으로 매칭해야 합니다)
# 여기서는 간단히 모델 예측 후 값을 추출합니다.
scenario_23 = temp_df[temp_df['Year'] == 2023].copy()
scenario_24 = temp_df[temp_df['Year'] == 2024].copy()
scenario_25 = temp_df[temp_df['Year'] == 2025].copy() # 25년 데이터가 구축되어 있다고 가정

pred_23 = model.predict(scenario_23[features])
pred_24 = model.predict(scenario_24[features])
pred_25 = model.predict(scenario_25[features])

# ---------------------------------------------------------
# 6. 형님의 아이디어: "예측된 공급량의 3년 평균" 산출
# ---------------------------------------------------------
# 날짜(월/일)가 동일하도록 정렬되었다고 가정하고 데이터프레임 구성
# (실제 데이터에선 2월 29일 윤년 처리를 위해 2월 28일까지만 쓰거나 별도 처리 필요)
min_len = min(len(pred_23), len(pred_24), len(pred_25))

result_df = pd.DataFrame({
    'Month': scenario_23['Month'].values[:min_len],
    'Day': scenario_23['Day'].values[:min_len],
    'Pred_Supply_23': pred_23[:min_len],
    'Pred_Supply_24': pred_24[:min_len],
    'Pred_Supply_25': pred_25[:min_len]
})

# ★ 최종 예상 공급량 = (23년 예측공급량 + 24년 예측공급량 + 25년 예측공급량) / 3
result_df['Final_Expected_Supply'] = result_df[['Pred_Supply_23', 'Pred_Supply_24', 'Pred_Supply_25']].mean(axis=1)

print("=== 최종 공급량 시뮬레이션 결과 (미리보기) ===")
print(result_df.head(10))

# 최종 결과 저장
# result_df.to_csv('최종_공급량_시뮬레이션결과.csv', index=False, encoding='utf-8-sig')
