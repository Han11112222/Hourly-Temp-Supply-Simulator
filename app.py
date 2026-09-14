import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
st.markdown('<hr style="margin-top:0.5rem; margin-bottom:1.5rem; border:none; border-top:3px solid #1a3c5e;">',
            unsafe_allow_html=True)

# ==========================================
# 공통: 범례 클릭으로 라인을 껐다 켰다 할 수 있는 인터랙티브 차트
# (st.line_chart는 범례 클릭 토글을 지원하지 않아 Plotly로 통일)
# ==========================================
LINE_COLORS = {
    '실제_공급량합계':     "#1f4e9c",
    '방법1_예측(정밀)':    "#2ecc71",
    '방법2_예측(단순)':    "#f39c12",
    '냉방용_판매량':       "#1f4e9c",
    '예측_판매량':         "#66b2ff",
    '예측_판매량_v2':      "#e74c3c",
    '예측_판매량_v3':      "#8e44ad",
    '판매량_계획':         "#f1948a",
}

SERIES_LABELS = {
    '냉방용_판매량':  '실적',
    '예측_판매량':    '기존 단일 3차식',
    '예측_판매량_v2': '분리·3차식(참고)',
    '예측_판매량_v3': '분리·2차식',
    '판매량_계획':    '판매량 계획',
}



def render_line_chart(df, x_col, y_cols, height=420, title=None):
    """
    범례를 클릭하면 해당 라인을 껐다 켰다 할 수 있는 인터랙티브 라인차트.
    df: x_col을 포함한 DataFrame (set_index 하지 않은 상태로 전달)
    y_cols: 그릴 컬럼 이름 리스트 (df에 없는 컬럼은 자동으로 건너뜀)
    """
    fig = go.Figure()
    for col in y_cols:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], mode="lines+markers", name=col,
            line=dict(color=LINE_COLORS.get(col), width=2.2),
            marker=dict(size=5),
        ))
    layout_kwargs = dict(
        height=height,
        margin=dict(t=40 if title else 10, b=10, l=50, r=20),
        hovermode="x unified",
        yaxis=dict(rangemode="tozero", tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    if title:  # title=None을 그대로 넘기면 프론트엔드에서 "undefined"로 표시되는 문제 방지
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True, config=dict(displaylogo=False))


def render_r2_mae_card(col, label, r2, mae, delta_r2=None):
    """R²와 MAE를 같은 줄에 동일한 크기로 나란히 보여주는 카드. delta_r2가 있으면 그 아래 작게 개선폭 표시."""
    delta_html = ""
    if delta_r2 is not None:
        color = "#16a34a" if delta_r2 >= 0 else "#dc2626"
        arrow = "↑" if delta_r2 >= 0 else "↓"
        sign = "+" if delta_r2 >= 0 else ""
        delta_html = (f'<div style="font-size:0.85rem;color:{color};margin-top:4px;">'
                      f'{arrow} {sign}{delta_r2:.4f}</div>')
    col.markdown(f"""
<div style="font-size:0.8rem;color:#666;margin-bottom:2px;">{label}</div>
<div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;">
  <span style="font-size:1.9rem;font-weight:700;color:#1f2937;">{r2:.4f}</span>
  <span style="font-size:1.9rem;font-weight:700;color:#166534;background-color:#dcfce7;
               padding:0.05em 0.4em;border-radius:0.4em;">MAE {mae:,.0f}</span>
</div>
{delta_html}
""", unsafe_allow_html=True)


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
# [신규] 3. 냉방용 사용량 분석
#   - 기온: 일별 기온 시트 → 검침기간(전월16일~당월15일) 평균 (일평균기온 기준, 시간대별 정밀 방식 아님)
#   - 판매량: 판매량 실적 시트 '냉방용' 컬럼
#   - 모델: 검침기온 → 냉방용 판매량, 3차 다항식(Poly-3) 단일 모델
# ==========================================

SALES_SHEET_URL = "https://docs.google.com/spreadsheets/d/1-8RIPIkjnVXxoh5QJs6598nnHkWOGmrO655jr3b3g04/export?format=csv&gid=0"
PLAN_SHEET_URL = "https://docs.google.com/spreadsheets/d/1zu2R21_P6z6yCeWz7yX1K6IYhj541hcr3IvCAaHLEQ8/export?format=csv&gid=0"

# ── Ver2(동절기/하절기 분리 모델)용 기준온도 — 기존 HDD/CDD 기준과 동일 ──
WINTER_T = 18.0  # 검침기온 ≤ 18℃ → 동절기 모델 (HDD 기준온도)
SUMMER_T = 26.0  # 검침기온 ≥ 26℃ → 하절기 모델 (CDD 기준온도)


@st.cache_data
def load_daily_temp_for_cooling():
    """
    구글시트(13HrIz6O...)의 일자 단위 원본 기온을 그대로 로드한다.
    (load_monthly_avg_temp()는 이미 월평균으로 뭉개버리므로,
     검침기간 전월16~당월15 계산을 위해 일자 단위로 별도 로드)
    """
    sheet_url = "https://docs.google.com/spreadsheets/d/13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs/export?format=csv&gid=0"
    try:
        df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"❌ 일별기온 구글시트 로드 오류: {e}")
        st.stop()

    col_list = df.columns.tolist()
    date_cols = [c for c in col_list if '날짜' in c or 'date' in c.lower() or 'Date' in c]
    DATE_COL = date_cols[0] if date_cols else col_list[0]
    temp_cols = [c for c in col_list if '평균기온' in c] or \
                [c for c in col_list if '기온' in c or 'temp' in c.lower()]
    TEMP_COL = temp_cols[0] if temp_cols else col_list[1]

    df['Date'] = pd.to_datetime(df[DATE_COL], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day']   = df['Date'].dt.day
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce')
    df = df.dropna(subset=[TEMP_COL])
    return df[['Date', 'Year', 'Month', 'Day', TEMP_COL]].rename(columns={TEMP_COL: 'Avg_Temp'})


def compute_meter_reading_temp(daily_df):
    """
    검침기간 평균기온 = 전월16일~말일 + 당월1일~15일 평균 (일평균기온 기준, 정밀 시간대 아님).
    반환: DataFrame(Year, Month, 검침기온)
    """
    rows = []
    for (y, m), _ in daily_df.groupby(['Year', 'Month']):
        cur_half = daily_df[(daily_df['Year'] == y) & (daily_df['Month'] == m) &
                             (daily_df['Day'] <= 15)]['Avg_Temp']
        py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
        prev_half = daily_df[(daily_df['Year'] == py) & (daily_df['Month'] == pm) &
                              (daily_df['Day'] >= 16)]['Avg_Temp']
        combined = pd.concat([prev_half, cur_half]).dropna()
        if len(combined) >= 5:
            rows.append({'Year': int(y), 'Month': int(m), '검침기온': combined.mean()})
    return pd.DataFrame(rows)


@st.cache_data
def load_cooling_sales():
    """판매량 실적 구글시트 — '냉방용' 컬럼 로드."""
    try:
        df = pd.read_csv(SALES_SHEET_URL)
    except Exception as e:
        st.error(f"❌ 판매량 구글시트 로드 오류: {e}")
        st.stop()

    col_list = df.columns.tolist()
    cooling_col = None
    for c in col_list:
        if '냉방' in c:
            cooling_col = c; break
    if cooling_col is None:
        st.error("판매량 시트에서 '냉방용' 컬럼을 찾을 수 없습니다.")
        st.stop()

    year_col  = '연' if '연' in col_list else ('Year' if 'Year' in col_list else col_list[1])
    month_col = '월' if '월' in col_list else ('Month' if 'Month' in col_list else col_list[2])

    out = df.rename(columns={year_col: 'Year', month_col: 'Month'})[['Year', 'Month', cooling_col]].copy()
    out[cooling_col] = pd.to_numeric(
        out[cooling_col].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')
    out['Year']  = pd.to_numeric(out['Year'], errors='coerce')
    out['Month'] = pd.to_numeric(out['Month'], errors='coerce')
    out = out.dropna(subset=['Year', 'Month', cooling_col])
    out['Year']  = out['Year'].astype(int)
    out['Month'] = out['Month'].astype(int)
    out = out[out[cooling_col] > 0].reset_index(drop=True)
    return out.rename(columns={cooling_col: '냉방용_판매량'})


@st.cache_data
def load_cooling_plan():
    """
    '상품별판매량 계획' 구글시트 — '냉방용' 컬럼(기존 계획값) 로드.
    로드 실패/컬럼 미탐지 시 None을 반환하며, 호출부에서 계획 비교 없이 진행하도록 처리한다.
    """
    try:
        df = pd.read_csv(PLAN_SHEET_URL)
    except Exception as e:
        st.sidebar.warning(f"⚠️ 판매량 계획 시트 로드 실패: {e} (계획 비교 생략)")
        return None

    col_list = df.columns.tolist()
    plan_col = None
    for c in col_list:
        if '냉방' in c:
            plan_col = c; break
    if plan_col is None:
        st.sidebar.warning("판매량 계획 시트에서 '냉방용' 컬럼을 찾을 수 없어 계획 비교를 생략합니다.")
        return None

    year_col  = '연' if '연' in col_list else ('Year' if 'Year' in col_list else col_list[1])
    month_col = '월' if '월' in col_list else ('Month' if 'Month' in col_list else col_list[2])

    out = df.rename(columns={year_col: 'Year', month_col: 'Month'})[['Year', 'Month', plan_col]].copy()
    out[plan_col] = pd.to_numeric(
        out[plan_col].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')
    out['Year']  = pd.to_numeric(out['Year'], errors='coerce')
    out['Month'] = pd.to_numeric(out['Month'], errors='coerce')
    out = out.dropna(subset=['Year', 'Month', plan_col])
    out['Year']  = out['Year'].astype(int)
    out['Month'] = out['Month'].astype(int)
    return out.rename(columns={plan_col: '판매량_계획'})


def fit_piecewise_seasonal_models(train_df, x_col='검침기온', y_col='냉방용_판매량', degree=3):
    """
    검침기온 기준 동절기(≤WINTER_T)/하절기(≥SUMMER_T) 데이터를 각각 나눠
    별도의 다항식 모델을 학습한다. (이중계상 방지를 위해 중간구간 데이터는 학습에서 제외,
    예측 시 18℃/26℃ 경계값을 선형보간하여 연결)
    """
    winter_data = train_df[train_df[x_col] <= WINTER_T]
    summer_data = train_df[train_df[x_col] >= SUMMER_T]

    models = {'winter': None, 'summer': None}
    min_pts = degree + 1
    if len(winter_data) >= min_pts:
        mw = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        mw.fit(winter_data[[x_col]], winter_data[y_col])
        models['winter'] = mw
    if len(summer_data) >= min_pts:
        ms = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        ms.fit(summer_data[[x_col]], summer_data[y_col])
        models['summer'] = ms
    return models, winter_data, summer_data


def predict_piecewise_seasonal(models, x_values):
    """
    x_values(검침기온 배열)에 대해:
      x <= WINTER_T        → 동절기 모델 예측
      x >= SUMMER_T         → 하절기 모델 예측
      WINTER_T < x < SUMMER_T → 두 모델의 경계값(18℃/26℃ 지점 예측)을 선형보간
    """
    x_arr = np.asarray(x_values, dtype=float)
    mw, ms = models.get('winter'), models.get('summer')
    w_at_boundary = float(mw.predict([[WINTER_T]])[0]) if mw is not None else None
    s_at_boundary = float(ms.predict([[SUMMER_T]])[0]) if ms is not None else None

    preds = np.full_like(x_arr, np.nan, dtype=float)
    for i, x in enumerate(x_arr):
        if np.isnan(x):
            continue
        if x <= WINTER_T:
            preds[i] = float(mw.predict([[x]])[0]) if mw is not None else np.nan
        elif x >= SUMMER_T:
            preds[i] = float(ms.predict([[x]])[0]) if ms is not None else np.nan
        else:
            if w_at_boundary is not None and s_at_boundary is not None:
                frac = (x - WINTER_T) / (SUMMER_T - WINTER_T)
                preds[i] = w_at_boundary * (1 - frac) + s_at_boundary * frac
            elif w_at_boundary is not None:
                preds[i] = w_at_boundary
            elif s_at_boundary is not None:
                preds[i] = s_at_boundary
    return preds


def poly_eq_str(coefs, intercept):
    """
    PolynomialFeatures(degree=n, include_bias=False) 계수 배열(coefs, 오름차순: x, x², x³...)과
    절편(intercept)을 받아 차수에 상관없이 "y = ax^n + ... + c" 형태 문자열을 만든다.
    """
    n = len(coefs)
    parts = []
    for power in range(n, 0, -1):
        c = coefs[power - 1]
        parts.append(f"{c:+.2f}x^{power}" if power > 1 else f"{c:+.2f}x")
    parts.append(f"{intercept:+.0f}")
    eq = " ".join(parts)
    if eq.startswith("+"):
        eq = eq[1:]
    return f"y = {eq}"


def _dynamic_fmt(df, x_col):
    """df의 x_col을 제외한 모든 컬럼에 대해, '오차율'이 들어간 컬럼은 %, 나머지는 천단위 콤마로 포맷."""
    fmt = {}
    for c in df.columns:
        if c == x_col:
            continue
        fmt[c] = "{:.1f}%" if '오차율' in c else "{:,.0f}"
    return fmt


def _build_diff_table(df, x_col, target_col, selected_cols):
    """
    df에서 x_col + selected_cols(원본값 컬럼)만 뽑아 표를 만들고,
    target_col이 selected_cols에 포함돼 있으면 나머지 선택 시리즈마다
    target 대비 '_차이' / '_오차율(%)' 컬럼을 자동으로 추가한다.
    """
    cols = [c for c in selected_cols if c in df.columns]
    out = df[[x_col] + cols].copy()
    if target_col in cols:
        for c in cols:
            if c == target_col:
                continue
            out[f'{c}_차이'] = out[c] - out[target_col]
            with np.errstate(divide='ignore', invalid='ignore'):
                out[f'{c}_오차율(%)'] = np.where(
                    out[target_col] != 0, out[f'{c}_차이'] / out[target_col] * 100, np.nan)
    return out


def render_cooling_analysis():
    st.header("🧊 냉방용 사용량 분석 심화ver")
    st.markdown("- 전월16일부터 당월15일까지의 실제기온 평균 적용 (세 가지 모델 모두 공통)")

    with st.spinner("냉방용 데이터를 불러오는 중입니다..."):
        daily_temp_df = load_daily_temp_for_cooling()
        meter_temp_df = compute_meter_reading_temp(daily_temp_df)
        sales_df = load_cooling_sales()
        plan_df = load_cooling_plan()  # None일 수 있음 (로드 실패/컬럼 미탐지 시 계획 비교 생략)
        merged_cool = pd.merge(meter_temp_df, sales_df, on=['Year', 'Month'], how='inner')
        merged_cool['Year_Month'] = merged_cool.apply(
            lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1)

    if merged_cool.empty:
        st.warning("실제기온과 판매량 데이터의 겹치는 기간이 없습니다.")
        st.stop()

    TARGET = '냉방용_판매량'
    all_years_cool = sorted(merged_cool['Year'].unique())

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🧊 냉방용 분석 설정**")
    train_years_c = st.sidebar.multiselect(
        "1. AI 학습 연도 선택 (냉방용)", options=all_years_cool,
        default=all_years_cool, key="cool_train_years")
    eval_years_c = st.sidebar.multiselect(
        "2. 과거 적합도 검증 연도 (냉방용)", options=all_years_cool,
        default=all_years_cool[-2:], key="cool_eval_years")
    max_year_c = int(merged_cool['Year'].max())
    future_years_c = st.sidebar.multiselect(
        "3. 미래 시나리오 추정 연도 (냉방용)",
        options=list(range(max_year_c + 1, max_year_c + 6)),
        default=[max_year_c + 1, max_year_c + 2], key="cool_future_years")
    y_years_c = st.sidebar.slider(
        "4. 미래 예측기온 추정 기준 (최근 Y년 평균, 냉방용)",
        min_value=1, max_value=10, value=3, step=1, key="cool_y_years")
    sim_base_years_c = list(range(max_year_c - y_years_c + 1, max_year_c + 1))

    if not train_years_c or not eval_years_c:
        st.warning("👈 좌측 패널에서 냉방용 학습/검증 연도를 선택해주세요.")
        st.stop()

    train_df_c = merged_cool[merged_cool['Year'].isin(train_years_c)]
    x_train_c = train_df_c[['검침기온']]
    y_train_c = train_df_c[TARGET]

    # ══════════════════════════════════════════
    # 모델 설명 (요약)
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown(f"""
**1. 일반적인 3차 다항식 적용**
(단점: 겨울·여름 두 번 꺾이는 패턴을 한 곡선에 억지로 담다 보니 하절기에서 과대예측(overshoot) 발생)

**2. 동절기/하절기 분리 (HDD {WINTER_T:.0f}℃ / CDD {SUMMER_T:.0f}℃ 적용)**
실제기온 {WINTER_T:.0f}℃ 이하는 동절기 모델, {SUMMER_T:.0f}℃ 이상은 하절기 모델로 각각 학습하고,
그 사이 구간은 두 모델의 경계값을 선형보간해 연결 (중복계상 방지)

**3. 추가 모델 (2차식)**
하절기는 학습 표본이 적어 3차식은 계수가 불안정해지고 과적합 위험이 있어, 동절기·하절기 모두
2차식으로 낮춰 예측을 안정화한 모델을 추가로 제공합니다 (아래 R² 비교로 개선 효과 확인 가능)
""")

    # 기준모델(단일 3차식) — 비교 지표용으로만 사용, 별도 섹션은 만들지 않음
    model_base = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    model_base.fit(x_train_c, y_train_c)

    # 분리모델(3차식) — 2차식 채택 근거 비교용 (표/차트에는 노출하지 않고 R² 지표에만 사용)
    models_cubic, _, _ = fit_piecewise_seasonal_models(train_df_c, x_col='검침기온', y_col=TARGET, degree=3)

    # 최종모델(2차식, 동절기/하절기 분리) — 이 화면에서 실제로 채택해 보여주는 모델
    models_final, winter_data_f, summer_data_f = fit_piecewise_seasonal_models(
        train_df_c, x_col='검침기온', y_col=TARGET, degree=2)

    if models_final['winter'] is None or models_final['summer'] is None:
        st.warning(
            f"동절기(≤{WINTER_T:.0f}℃, n={len(winter_data_f)}) 또는 "
            f"하절기(≥{SUMMER_T:.0f}℃, n={len(summer_data_f)}) 학습 데이터가 3건 미만이라 "
            "모델을 만들 수 없습니다. 학습 연도를 늘려주세요."
        )
        st.stop()

    cw = models_final['winter'].named_steps['linearregression'].coef_
    iw = models_final['winter'].named_steps['linearregression'].intercept_
    cs = models_final['summer'].named_steps['linearregression'].coef_
    isu = models_final['summer'].named_steps['linearregression'].intercept_
    r2_w = r2_score(winter_data_f[TARGET], models_final['winter'].predict(winter_data_f[['검침기온']]))
    r2_s = r2_score(summer_data_f[TARGET], models_final['summer'].predict(summer_data_f[['검침기온']]))

    col_w, col_s = st.columns(2)
    with col_w:
        st.info(f"""
**❄️ 동절기 모델 (실제기온 ≤ {WINTER_T:.0f}℃, n={len(winter_data_f)}, 2차식)**

학습 R² = {r2_w * 100:.2f}%

${poly_eq_str(cw, iw)}$
""")
    with col_s:
        st.info(f"""
**☀️ 하절기 모델 (실제기온 ≥ {SUMMER_T:.0f}℃, n={len(summer_data_f)}, 2차식)**

학습 R² = {r2_s * 100:.2f}%

${poly_eq_str(cs, isu)}$
""")
    st.caption(f"※ {WINTER_T:.0f}℃부터 {SUMMER_T:.0f}℃ 사이 구간은 두 모델의 경계값을 선형보간하여 연결(중복계상 방지) "
               f"· 기온 소스: 구글시트 일별 기온 → 검침기간(전월16일부터 당월15일까지) 평균 · 판매량 소스: 판매량 실적 시트 — 냉방용")

    with st.expander("🔎 실제기온 ↔ 냉방용 판매량 산점도 (학습 데이터)"):
        st.scatter_chart(train_df_c.rename(columns={'검침기온': '실제기온'}), x='실제기온', y=TARGET, height=380)

    # ══════════════════════════════════════════
    # 과거 적합도 검증
    # ══════════════════════════════════════════
    st.subheader("📊 과거 모델 적합도 검증 (냉방용)")
    eval_df_c = merged_cool[merged_cool['Year'].isin(eval_years_c)].copy()
    eval_df_c['예측_판매량'] = model_base.predict(eval_df_c[['검침기온']])
    eval_df_c['예측_판매량_v3'] = predict_piecewise_seasonal(models_final, eval_df_c['검침기온'].values)

    valid_eval = eval_df_c['예측_판매량_v3'].notna()
    r2_base_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량'])
    mae_base_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량'] - eval_df_c.loc[valid_eval, TARGET]))
    r2_final_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량_v3'])
    mae_final_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량_v3'] - eval_df_c.loc[valid_eval, TARGET]))

    has_cubic_split = models_cubic['winter'] is not None and models_cubic['summer'] is not None
    if has_cubic_split:
        eval_df_c['예측_판매량_v2'] = predict_piecewise_seasonal(models_cubic, eval_df_c['검침기온'].values)
        r2_cubic_eval = r2_score(eval_df_c.loc[valid_eval, TARGET], eval_df_c.loc[valid_eval, '예측_판매량_v2'])
        mae_cubic_eval = np.mean(np.abs(eval_df_c.loc[valid_eval, '예측_판매량_v2'] - eval_df_c.loc[valid_eval, TARGET]))

    monthly_eval_c = eval_df_c[['Year_Month', 'Year', 'Month', TARGET, '예측_판매량', '예측_판매량_v3']].copy()
    if has_cubic_split:
        monthly_eval_c['예측_판매량_v2'] = eval_df_c['예측_판매량_v2']

    has_plan_eval = False
    if plan_df is not None:
        monthly_eval_c = monthly_eval_c.merge(plan_df, on=['Year', 'Month'], how='left')
        has_plan_eval = monthly_eval_c['판매량_계획'].notna().any()

    all_series_eval = [TARGET, '예측_판매량'] + (['예측_판매량_v2'] if has_cubic_split else []) \
        + ['예측_판매량_v3'] + (['판매량_계획'] if has_plan_eval else [])

    # MAE가 가장 낮은(=가장 적합한) 모델에 자동으로 ✅ 표시
    mae_by_model = {"기존 단일 3차식": mae_base_eval, "분리·2차식": mae_final_eval}
    if has_cubic_split:
        mae_by_model["분리·3차식 (참고)"] = mae_cubic_eval
    best_label = min(mae_by_model, key=mae_by_model.get)
    label_base = "✅ 기존 단일 3차식" if best_label == "기존 단일 3차식" else "기존 단일 3차식"
    label_cubic = "✅ 분리·3차식 (참고)" if best_label == "분리·3차식 (참고)" else "분리·3차식 (참고)"
    label_final = "✅ 분리·2차식" if best_label == "분리·2차식" else "분리·2차식"

    mcols = st.columns(3 if has_cubic_split else 2)
    render_r2_mae_card(mcols[0], label_base, r2_base_eval, mae_base_eval)
    if has_cubic_split:
        render_r2_mae_card(mcols[1], label_cubic, r2_cubic_eval, mae_cubic_eval,
                           delta_r2=r2_cubic_eval - r2_base_eval)
        render_r2_mae_card(mcols[2], label_final, r2_final_eval, mae_final_eval,
                           delta_r2=r2_final_eval - r2_base_eval)
    else:
        render_r2_mae_card(mcols[1], label_final, r2_final_eval, mae_final_eval,
                           delta_r2=r2_final_eval - r2_base_eval)

    # 차트는 항상 전체 시리즈 표시 — 플롯리 자체 범례 클릭으로 라인 표시/숨김
    render_line_chart(monthly_eval_c, 'Year_Month', all_series_eval, height=420)

    # 아래 선택 위젯은 표(연도별/월별)에만 반영됨 (차트에는 영향 없음)
    st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영됩니다)")
    selected_eval = st.multiselect(
        "표시할 시리즈", options=all_series_eval, default=all_series_eval,
        format_func=lambda c: SERIES_LABELS.get(c, c), key="eval_series_select")
    if not selected_eval:
        st.info("표시할 항목을 1개 이상 선택해주세요. 우선 전체 항목을 표시합니다.")
        selected_eval = all_series_eval

    yearly_agg_eval = monthly_eval_c.groupby('Year')[all_series_eval].sum().reset_index()
    yearly_table_eval = _build_diff_table(yearly_agg_eval, 'Year', TARGET, selected_eval)
    st.markdown("**📆 연도별 실적 대비 차이 요약**")
    st.dataframe(yearly_table_eval.style.format(_dynamic_fmt(yearly_table_eval, 'Year')),
                 use_container_width=True, hide_index=True)

    monthly_table_eval = _build_diff_table(monthly_eval_c, 'Year_Month', TARGET, selected_eval)
    st.markdown("**🗂️ 월별 상세 비교**")
    st.dataframe(monthly_table_eval.style.format(_dynamic_fmt(monthly_table_eval, 'Year_Month')),
                 use_container_width=True, hide_index=True)

    dl_eval1, dl_eval2 = st.columns(2)
    with dl_eval1:
        csv_yearly_eval = yearly_table_eval.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 연도별 요약 다운로드", data=csv_yearly_eval,
                           file_name="냉방용_연도별요약.csv", mime="text/csv", key="dl_eval_yearly")
    with dl_eval2:
        csv_monthly_eval = monthly_table_eval.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 월별 상세 다운로드", data=csv_monthly_eval,
                           file_name="냉방용_과거적합도_검증리포트.csv", mime="text/csv", key="dl_eval_monthly")

    # ══════════════════════════════════════════
    # 미래 시나리오
    # ══════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔮 미래 냉방용 판매량 추정 시나리오")

    if future_years_c:
        hist_temp_c = meter_temp_df[meter_temp_df['Year'].isin(sim_base_years_c)]
        sim_month_temp_c = hist_temp_c.groupby('Month')['검침기온'].mean().reset_index()

        future_rows = []
        for y in future_years_c:
            for m in range(1, 13):
                t = sim_month_temp_c.loc[sim_month_temp_c['Month'] == m, '검침기온']
                if len(t) > 0:
                    future_rows.append({'Year': y, 'Month': m, '검침기온': float(t.values[0])})
        future_df_c = pd.DataFrame(future_rows)
        future_df_c['예측_판매량'] = model_base.predict(future_df_c[['검침기온']])
        future_df_c['예측_판매량_v3'] = predict_piecewise_seasonal(models_final, future_df_c['검침기온'].values)
        if has_cubic_split:
            future_df_c['예측_판매량_v2'] = predict_piecewise_seasonal(models_cubic, future_df_c['검침기온'].values)
        future_df_c['Year_Month'] = future_df_c.apply(
            lambda r: f"{int(r['Year'])}-{int(r['Month']):02d}", axis=1)

        # 실제 실적이 있으면(예: 최근 진행 중인 연도) 함께 표시
        future_df_c = pd.merge(future_df_c, sales_df, on=['Year', 'Month'], how='left')
        has_actual = TARGET in future_df_c.columns and future_df_c[TARGET].notna().any()

        # 판매량 계획(기존 계획, 상품별판매량 계획 시트) 병합
        has_plan_future = False
        if plan_df is not None:
            future_df_c = pd.merge(future_df_c, plan_df, on=['Year', 'Month'], how='left')
            has_plan_future = future_df_c['판매량_계획'].notna().any()

        st.caption(f"미래 예측기온 추정: 최근 {y_years_c}개년"
                   f"({min(sim_base_years_c)}~{max(sim_base_years_c)}) 동월 실제기온 평균 사용")

        agg_cols_fut = ['예측_판매량'] + (['예측_판매량_v2'] if has_cubic_split else []) \
            + ['예측_판매량_v3'] + ([TARGET] if has_actual else []) \
            + (['판매량_계획'] if has_plan_future else [])

        # 차트는 항상 전체 시리즈 표시 — 플롯리 자체 범례 클릭으로 라인 표시/숨김
        render_line_chart(future_df_c, 'Year_Month', agg_cols_fut, height=420)

        # 아래 선택 위젯은 표(연도별/월별)에만 반영됨 (차트에는 영향 없음)
        st.markdown("**📌 표에 표시할 항목 선택** (아래 연도별·월별 표에만 반영됩니다)")
        selected_fut = st.multiselect(
            "표시할 시리즈", options=agg_cols_fut, default=agg_cols_fut,
            format_func=lambda c: SERIES_LABELS.get(c, c), key="future_series_select")
        if not selected_fut:
            st.info("표시할 항목을 1개 이상 선택해주세요. 우선 전체 항목을 표시합니다.")
            selected_fut = agg_cols_fut

        yearly_future_c = future_df_c.groupby('Year')[agg_cols_fut].sum().reset_index()
        yearly_future_c = _build_diff_table(
            yearly_future_c, 'Year', TARGET if has_actual else '예측_판매량_v3', selected_fut)
        st.markdown("**📆 연도별 시나리오 합산**")
        st.dataframe(yearly_future_c.style.format(_dynamic_fmt(yearly_future_c, 'Year')),
                     use_container_width=True, hide_index=True)

        show_cols = ['Year_Month', '검침기온'] + selected_fut
        disp_future = future_df_c[show_cols].rename(columns={'검침기온': '예측기온'})
        fmt_disp_future = _dynamic_fmt(disp_future, 'Year_Month')
        fmt_disp_future['예측기온'] = "{:.1f}℃"
        st.markdown("**🗂️ 월별 시나리오**")
        st.dataframe(disp_future.style.format(fmt_disp_future, na_rep='-'),
                     use_container_width=True, hide_index=True)

        csv_future_c = disp_future.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 냉방용 미래 시나리오 다운로드", data=csv_future_c,
                           file_name="냉방용_미래시나리오.csv", mime="text/csv")
    else:
        st.info("좌측에서 미래 시나리오 추정 연도를 선택하면 결과가 표시됩니다.")



# ==========================================
# 2. 좌측 사이드바: 컨트롤 패널
# ==========================================
st.sidebar.header("⚙️ 시뮬레이션 설정 패널")

analysis_mode = st.sidebar.radio(
    "📊 분석 대상 선택",
    options=["1. 전체 공급량 분석", "2. 개별난방용 공급량 분석", "3. 냉방용 사용량 분석"],
    index=0
)

# ★ 신규 추가: 3번 선택 시 기존 파이프라인(공급량 분석)은 건드리지 않고 바로 분기
if analysis_mode == "3. 냉방용 사용량 분석":
    render_cooling_analysis()
    st.stop()

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
render_line_chart(monthly_eval, 'Year_Month', chart_cols_eval, height=550)
st.caption("범례 클릭 시 라인 표시/숨김")

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
render_line_chart(monthly_future, 'Year_Month', available_chart_cols, height=550)
st.caption("범례 클릭 시 라인 표시/숨김")

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
