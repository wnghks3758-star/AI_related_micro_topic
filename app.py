import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import os
import glob

# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 상태 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="AI 뉴스 인사이트 대시보드", page_icon="🌐", layout="wide")

st.title("🌐 글로벌 AI 뉴스 인사이트 대시보드")
st.markdown("수집된 뉴스 기사들을 토픽별로 분석하여 카테고리 기반 핵심 인사이트를 제공합니다.")
st.markdown("---")

# 검색어를 저장할 Session State 초기화 (버튼 클릭과 검색창을 동기화하기 위함)
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

def set_search_keyword(keyword):
    """키워드 버튼 클릭 시 검색창 상태를 업데이트하는 콜백 함수"""
    st.session_state.search_query = keyword

# ---------------------------------------------------------
# 2. 데이터 자동 로드 및 기간 선택 (GitHub 배포 환경 대응)
# ---------------------------------------------------------

# 💡 Parquet 파일들이 저장된 폴더 경로 (GitHub 구조에 맞게 수정 가능)
DATA_DIR = "data/result"

@st.cache_data
def get_available_periods(data_dir):
    """지정된 폴더에서 Parquet 파일 목록을 읽어와 (보여줄 기간 이름 : 실제 파일 경로) 딕셔너리를 만듭니다."""
    file_map = {}
    
    # 폴더가 존재하는지 확인
    if not os.path.exists(data_dir):
        return file_map
        
    # 폴더 안의 파일들을 하나씩 확인
    for file_name in os.listdir(data_dir):
        if file_name.startswith("Micro_Topics_") and file_name.endswith(".parquet"):
            # 파일명에서 날짜 부분만 추출 (예: Micro_Topics_2026-03-30_to_2026-04-12.parquet)
            date_str = file_name.replace("Micro_Topics_", "").replace(".parquet", "")
            
            try:
                # "2026-03-30_to_2026-04-12" -> "2026-03-30", "2026-04-12" 분리
                start_date, end_date = date_str.split("_to_")
                label = f"{start_date} ~ {end_date}"
                
                # 파일 경로 저장
                file_map[label] = os.path.join(data_dir, file_name)
            except ValueError:
                # 파일 이름 규칙이 맞지 않는 경우 무시
                pass
                
    # 최신 날짜가 위로 오도록 정렬하여 반환
    return dict(sorted(file_map.items(), reverse=True))

# 폴더 읽기 실행
file_map = get_available_periods(DATA_DIR)

# 파일이 하나도 없는 경우 처리
if not file_map:
    st.error(f"'{DATA_DIR}' 폴더에서 분석 가능한 Parquet 파일을 찾을 수 없습니다. GitHub 저장소 경로를 확인해 주세요.")
    st.stop()

# 사이드바에 기간 선택 드롭다운 생성
selected_period = st.sidebar.selectbox("📅 분석할 기간을 선택하세요", list(file_map.keys()))

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

# 선택된 기간의 파일 로드
try:
    df = load_data(file_map[selected_period])
    # 성공 메시지는 3초 후 사라지도록 처리하면 화면이 더 깔끔합니다.
    st.success(f"'{selected_period}' 기간의 데이터가 로드되었습니다!", icon="✅")
except Exception as e:
    st.error(f"파일을 읽는 중 에러가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 사이드바: 카테고리 필터링 및 검색창
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 및 검색")

# 카테고리 선택
category_list = sorted(df['카테고리'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox(
    "📂 카테고리 선택", 
    ["전체보기"] + category_list
)

# 💡 Session State의 'search_query' 키와 연동된 검색창
search_input = st.sidebar.text_input(
    "🔎 검색어 입력 (제목, 요약, 키워드 포함)", 
    key="search_query" # 이 부분이 핵심입니다!
)

# 1차 필터링 (카테고리)
if selected_category == "전체보기":
    cat_df = df
else:
    cat_df = df[df['카테고리'] == selected_category]

# ---------------------------------------------------------
# 4. 메인 화면: 주요 키워드 추출 및 버튼 UI 배치
# ---------------------------------------------------------
# 현재 카테고리 데이터에서 모든 키워드 추출 및 빈도 계산
all_keywords = []
for kw_str in cat_df['키워드'].dropna():
    # 콤마로 분리하고 양옆 공백 제거, 빈 문자열 제외
    kws = [k.strip() for k in str(kw_str).split(',') if k.strip()]
    all_keywords.extend(kws)

# 가장 많이 등장한 상위 10개 키워드 추출
top_10_keywords = [kw for kw, count in Counter(all_keywords).most_common(10)]

if top_10_keywords:
    st.markdown("##### 🔥 현재 카테고리 주요 키워드 Top 10")
    st.caption("버튼을 클릭하면 해당 키워드가 포함된 리포트만 자동으로 필터링됩니다.")
    
    # 키워드 버튼을 5개씩 2줄로 예쁘게 배치
    for i in range(0, len(top_10_keywords), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(top_10_keywords):
                kw = top_10_keywords[i + j]
                # 버튼 클릭 시 set_search_keyword 함수 실행
                col.button(kw, on_click=set_search_keyword, args=(kw,), use_container_width=True)
    st.markdown("---")

# ---------------------------------------------------------
# 5. 검색어 2차 필터링 및 결과 출력
# ---------------------------------------------------------
filtered_df = cat_df

# 검색어(search_input)가 존재할 경우에만 필터링 적용
if search_input:
    filtered_df = cat_df[
        cat_df['제목'].str.contains(search_input, case=False, na=False) |
        cat_df['summary'].str.contains(search_input, case=False, na=False) |
        cat_df['키워드'].str.contains(search_input, case=False, na=False)
    ]

st.caption(f"검색 결과: 총 **{len(filtered_df)}**개의 토픽이 발견되었습니다.")
if search_input:
    st.success(f"💡 '**{search_input}**'에 대한 검색 결과입니다. (검색 취소는 좌측 검색창을 비워주세요)")
st.write("")

# 최종 결과 렌더링
for index, row in filtered_df.iterrows():
    expander_title = f"📌 {row['제목']} (키워드: {row['키워드']})"
    
    with st.expander(expander_title):
        st.markdown("#### 📝 핵심 인사이트 요약")
        st.markdown(row['summary'])
        
        st.markdown("---")
        st.markdown("#### 📑 관련 기사 출처")
        
        ref_data = row['출처_기사']
        if isinstance(ref_data, np.ndarray):
            ref_data = ref_data.tolist()
            
        if isinstance(ref_data, list) and len(ref_data) > 0:
            for ref in ref_data:
                title = ref.get('제목', '제목 없음')
                url = ref.get('URL', '#')
                st.markdown(f"- [{title}]({url})")
        else:
            st.write("관련 기사 정보가 없습니다.")