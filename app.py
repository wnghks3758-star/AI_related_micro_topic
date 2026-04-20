import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(page_title="AI 뉴스 인사이트 대시보드", page_icon="🌐", layout="wide")

st.title("🌐 글로벌 AI 뉴스 인사이트 대시보드")
st.markdown("수집된 뉴스 기사들을 토픽별로 분석하여 카테고리 기반 핵심 인사이트를 제공합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 2. 데이터 업로드 및 로드 (클라우드 환경 대응)
# ---------------------------------------------------------
# 사이드바나 메인 화면에 파일 업로더 생성
uploaded_file = st.file_uploader("📂 분석할 Parquet 결과 파일을 업로드해 주세요", type=['parquet'])

# 업로드된 파일을 읽어오는 함수 (캐싱 적용)
@st.cache_data
def load_uploaded_data(file_buffer):
    return pd.read_parquet(file_buffer)

# 파일이 업로드되지 않은 상태면 안내 문구를 띄우고 아래 코드 실행 중단
if uploaded_file is None:
    st.info("⬆️ 상단에 Parquet 파일을 업로드하시면 대시보드가 활성화됩니다.")
    st.stop()

# 파일이 업로드되면 데이터 로드 실행
try:
    df = load_uploaded_data(uploaded_file)
    st.success("데이터가 성공적으로 로드되었습니다!")
except Exception as e:
    st.error(f"파일을 읽는 중 에러가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 사이드바 필터링 및 검색 (사용자 인터페이스)
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 및 검색")

# 1. 카테고리 선택
category_list = sorted(df['카테고리'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox(
    "📂 카테고리 선택", 
    ["전체보기"] + category_list
)

# 2. 키워드 검색어 입력
search_query = st.sidebar.text_input("🔎 검색어 입력 (제목, 요약, 키워드 포함)", "").strip()

# [필터링 로직 시작]
# 카테고리 필터링 적용
if selected_category == "전체보기":
    filtered_df = df
else:
    filtered_df = df[df['카테고리'] == selected_category]

# 검색어 필터링 적용 (제목, summary, 키워드 중 하나라도 포함되면 노출)
if search_query:
    # 대소문자 구분 없이(case=False) 검색하며, NaN 값은 무시(na=False)하도록 설정
    filtered_df = filtered_df[
        filtered_df['제목'].str.contains(search_query, case=False, na=False) |
        filtered_df['summary'].str.contains(search_query, case=False, na=False) |
        filtered_df['키워드'].str.contains(search_query, case=False, na=False)
    ]

# 검색 결과 안내
st.caption(f"검색 결과: 총 **{len(filtered_df)}**개의 토픽이 발견되었습니다.")
if search_query:
    st.write(f"💡 '**{search_query}**'에 대한 검색 결과입니다.")
st.write("")

# ---------------------------------------------------------
# 4. 토픽 출력 및 하이퍼링크 처리
# ---------------------------------------------------------
for index, row in filtered_df.iterrows():
    # Expander(아코디언) 제목 설정: 직관적으로 키워드를 함께 노출
    expander_title = f"📌 {row['제목']} (키워드: {row['키워드']})"
    
    with st.expander(expander_title):
        
        # [요약본 노출 영역]
        st.markdown("#### 📝 핵심 인사이트 요약")
        st.markdown(row['summary'])
        
        st.markdown("---")
        
        # [출처 기사 노출 영역]
        st.markdown("#### 📑 관련 기사 출처")
        
        # Parquet 파일에서 리스트(또는 Numpy 배열) 형태로 불러와진 데이터를 안전하게 파싱
        ref_data = row['출처_기사']
        if isinstance(ref_data, np.ndarray):
            ref_data = ref_data.tolist()
            
        if isinstance(ref_data, list) and len(ref_data) > 0:
            for ref in ref_data:
                # 마크다운 문법: [텍스트](URL)을 사용하여 클릭 가능한 하이퍼링크 생성
                title = ref.get('제목', '제목 없음')
                url = ref.get('URL', '#')
                st.markdown(f"- [{title}]({url})")
        else:
            st.write("관련 기사 정보가 없습니다.")