import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import os
import pickle

# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 상태 초기화
# ---------------------------------------------------------
st.set_page_config(page_title="AI 뉴스 인사이트 대시보드", page_icon="🌐", layout="wide")

st.title("🌐 글로벌 AI 뉴스 인사이트 대시보드")
st.markdown("수집된 뉴스 기사들을 토픽별로 분석하여 카테고리 기반 핵심 인사이트를 제공합니다.")
st.markdown("---")

if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

def set_search_keyword(keyword):
    st.session_state.search_query = keyword

# ---------------------------------------------------------
# 2. 데이터 및 임베딩 사전 로드
# ---------------------------------------------------------
DATA_DIR = "data"

@st.cache_data
def get_available_periods(data_dir):
    file_map = {}
    if not os.path.exists(data_dir): return file_map
        
    for file_name in os.listdir(data_dir):
        if file_name.startswith("Micro_Topics_") and file_name.endswith(".parquet"):
            date_str = file_name.replace("Micro_Topics_", "").replace(".parquet", "")
            try:
                start_date, end_date = date_str.split("_to_")
                label = f"{start_date} ~ {end_date}"
                file_map[label] = os.path.join(data_dir, file_name)
            except ValueError:
                pass
    return dict(sorted(file_map.items(), reverse=True))

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

@st.cache_data
def load_embedding_dict(data_dir):
    """미리 만들어둔 키워드 임베딩 사전을 불러옵니다."""
    dict_path = os.path.join(data_dir, 'keyword_embeddings_dict.pkl')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            return pickle.load(f)
    return {}

file_map = get_available_periods(DATA_DIR)
if not file_map:
    st.error(f"'{DATA_DIR}' 폴더에서 분석 가능한 Parquet 파일을 찾을 수 없습니다.")
    st.stop()

selected_period = st.sidebar.selectbox("📅 분석할 기간을 선택하세요", list(file_map.keys()))
df = load_data(file_map[selected_period])
emb_dict = load_embedding_dict(DATA_DIR)

# ---------------------------------------------------------
# 3. 사이드바: 필터 및 고급 검색창
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 및 고급 검색")

category_list = sorted(df['카테고리'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("📂 카테고리 선택", ["전체보기"] + category_list)

# 고급 검색 안내 문구 추가
st.sidebar.caption("💡 **검색 팁:** 여러 단어를 검색할 때는 대문자로 `AND` 또는 `OR`을 사용해 보세요. (예: `APPLE AND AI`, `규제 OR 법안`)")
search_input = st.sidebar.text_input("🔎 검색어 입력", key="search_query").strip()

if selected_category == "전체보기":
    cat_df = df
else:
    cat_df = df[df['카테고리'] == selected_category]

# ---------------------------------------------------------
# 4. 메인 화면 1: 기본 토픽 키워드 Top 10 추출
# ---------------------------------------------------------
all_keywords = []
for kw_str in cat_df['키워드'].dropna():
    kws = [k.strip().upper() for k in str(kw_str).split(',') if k.strip()]
    all_keywords.extend(kws)

top_10_keywords = [kw for kw, count in Counter(all_keywords).most_common(10)]

if top_10_keywords and not search_input: # 검색어가 없을 때만 기본 Top 10 노출
    st.markdown("##### 🔥 현재 카테고리 주요 키워드 Top 10")
    for i in range(0, len(top_10_keywords), 5):
        cols = st.columns(5)
        for j, col in enumerate(cols):
            if i + j < len(top_10_keywords):
                kw = top_10_keywords[i + j]
                col.button(kw, key=f"top_{kw}", on_click=set_search_keyword, args=(kw,), use_container_width=True)
    st.markdown("---")

# ---------------------------------------------------------
# 5. 고급 검색 필터링 및 유사도 연관 검색어 추천
# ---------------------------------------------------------
filtered_df = cat_df

def apply_text_search(dataframe, keyword):
    """단일 키워드에 대한 필터링 조건(Mask)을 반환하는 함수"""
    return dataframe['제목'].str.contains(keyword, case=False, na=False) | \
           dataframe['summary'].str.contains(keyword, case=False, na=False) | \
           dataframe['키워드'].str.contains(keyword, case=False, na=False)

if search_input:
    search_upper = search_input.upper()
    
    # 1. AND / OR 논리 연산 필터링
    if " AND " in search_upper:
        keywords = [k.strip() for k in search_upper.split(" AND ") if k.strip()]
        mask = pd.Series([True] * len(cat_df), index=cat_df.index)
        for kw in keywords:
            mask = mask & apply_text_search(cat_df, kw)
        filtered_df = cat_df[mask]
        st.success(f"💡 조건 검색: **{' & '.join(keywords)}** 모두 포함된 결과입니다.")
        
    elif " OR " in search_upper:
        keywords = [k.strip() for k in search_upper.split(" OR ") if k.strip()]
        mask = pd.Series([False] * len(cat_df), index=cat_df.index)
        for kw in keywords:
            mask = mask | apply_text_search(cat_df, kw)
        filtered_df = cat_df[mask]
        st.success(f"💡 조건 검색: **{' | '.join(keywords)}** 중 하나라도 포함된 결과입니다.")
        
    else:
        # 2. 단일 키워드 일반 검색
        filtered_df = cat_df[apply_text_search(cat_df, search_input)]
        
        # 3. 임베딩 기반 연관 검색어(유사도) 추출 로직
        if emb_dict:
            if search_upper in emb_dict:
                query_emb = np.array(emb_dict[search_upper])
                similarities = []
                
                # 코사인 유사도 계산
                for vocab, vocab_emb in emb_dict.items():
                    if vocab == search_upper: continue
                    v_arr = np.array(vocab_emb)
                    sim = np.dot(query_emb, v_arr) / (np.linalg.norm(query_emb) * np.linalg.norm(v_arr))
                    similarities.append((vocab, sim))
                
                # 유사도 높은 순 정렬 후 Top 10 추출
                similarities.sort(key=lambda x: x[1], reverse=True)
                related_kws = [x[0] for x in similarities[:10]]
                
                st.markdown(f"##### 🔗 '{search_input}' 연관 키워드 (AI 유사도 기반)")
                for i in range(0, len(related_kws), 5):
                    cols = st.columns(5)
                    for j, col in enumerate(cols):
                        if i + j < len(related_kws):
                            kw = related_kws[i + j]
                            col.button(kw, key=f"rel_{kw}", on_click=set_search_keyword, args=(kw,), use_container_width=True)
                st.markdown("---")
            else:
                st.warning("⚠️ 입력하신 단어는 핵심 키워드 사전에 존재하지 않아 연관 검색어(유사도) 분석이 제공되지 않습니다. (일반 텍스트 검색 결과만 아래에 표시됩니다)")

st.caption(f"검색 결과: 총 **{len(filtered_df)}**개의 토픽이 발견되었습니다.")
st.write("")

# ---------------------------------------------------------
# 6. 최종 결과 렌더링
# ---------------------------------------------------------
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