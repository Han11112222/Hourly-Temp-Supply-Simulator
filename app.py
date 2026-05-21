import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 폰트 설정 (한글 및 마이너스 기호 깨짐 방지 - 윈도우 환경 맑은 고딕 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False 

# 2. 데이터 불러오기
# '합산기온.csv' 파일이 파이썬 스크립트와 같은 폴더에 있어야 합니다.
# (공공데이터 CSV는 보통 cp949 인코딩이 많아 우선 적용하고, 에러 시 utf-8로 읽도록 예외 처리했습니다)
print("데이터를 불러오는 중입니다...")
try:
    df = pd.read_csv('합산기온.csv', encoding='cp949', parse_dates=['일시'])
except UnicodeDecodeError:
    df = pd.read_csv('합산기온.csv', encoding='utf-8', parse_dates=['일시'])

# 3. 데이터 전처리 (월, 시간 추출)
print("데이터 전처리 중...")
# 기온 데이터가 있는 컬럼명을 확인하세요. (예: 사진상으로는 '기온' 대신 '2000'이나 별도 명칭일 수 있습니다)
# 여기서는 원본 CSV의 날짜 컬럼이 '일시', 온도 컬럼이 '기온'이라고 가정합니다.
# 만약 온도 컬럼 이름이 다르다면 아래 코드의 '기온' 부분을 실제 이름으로 바꿔주세요.
df['월'] = df['일시'].dt.month
df['시간'] = df['일시'].dt.hour

# 4. 피벗 테이블 생성 (월별, 시간대별 평균 기온 계산)
# 행(index)은 '월', 열(columns)은 '시간', 값(values)은 '기온'의 평균(mean)
pivot_df = df.pivot_table(index='월', columns='시간', values='기온', aggfunc='mean')

# 5. 히트맵 시각화
print("그래프를 그리는 중입니다...")
plt.figure(figsize=(16, 8))

# seaborn 히트맵 그리기
# cmap='coolwarm': 추울수록 파란색, 더울수록 빨간색으로 표시 (난방 수요 직관적 확인 가능)
# annot=True: 칸 안에 실제 평균 온도 숫자 표시
# fmt=".1f": 숫자를 소수점 첫째 자리까지만 표시
sns.heatmap(pivot_df, cmap='coolwarm', annot=True, fmt=".1f", linewidths=.5)

# 차트 디자인
plt.title('월별/시간대별 평균 기온 분포 (난방 수요 피크 타임 분석용)', fontsize=18, pad=20, fontweight='bold')
plt.xlabel('시간 (Hour)', fontsize=14)
plt.ylabel('월 (Month)', fontsize=14)

# y축 라벨(월)이 가로로 똑바로 보이도록 설정
plt.yticks(rotation=0) 

# 레이아웃을 깔끔하게 조정 후 출력
plt.tight_layout()
plt.show()

# (선택) 마케팅 회의 자료로 바로 쓸 수 있게 이미지 파일로 저장하려면 위 plt.show()를 지우고 아래 코드를 쓰세요.
# plt.savefig('월별_시간별_기온히트맵.png', dpi=300)
