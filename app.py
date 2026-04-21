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
def load_data(file_path, period_label):
    # Parquet 파일 로드
    df = pd.read_parquet(file_path)
    
    # 💡 [추가] period_label (예: "2026-03-30 ~ 2026-04-12")을 가공하여 컬럼으로 추가
    start_date, end_date = period_label.split(" ~ ")
    # 앞의 '20' 연도를 떼어내고 (인덱스 2부터 슬라이싱), 물결(~)로 연결
    short_date_format = f"{start_date[2:]}~{end_date[2:]}"
    
    # 데이터프레임에 '도출_기간' 컬럼 추가
    df['도출_기간'] = short_date_format
    
    return df

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

# 💡 [수정] load_data 호출 시 selected_period도 함께 넘겨줍니다.
df = load_data(file_map[selected_period], selected_period)
emb_dict = load_embedding_dict(DATA_DIR)

# ---------------------------------------------------------
# 3. 사이드바: 필터 및 고급 검색창
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 및 고급 검색")

# 카테고리 리스트 추출
category_list = ['기업 및 시장 동향', '기술 동향', '정책 및 규제 동향', '사회적 파급 효과', '기타']

# 💡 [수정] selectbox 대신 multiselect를 사용합니다.
# default를 빈 리스트([])로 두고, 비어있을 때 '전체보기'로 작동하도록 로직을 짭니다.
selected_categories = st.sidebar.multiselect(
    "📂 카테고리 선택", 
    options=category_list,
    default=[], 
    placeholder="전체보기 (클릭하여 카테고리 추가)"
)

# 고급 검색 안내 문구 추가
st.sidebar.caption("💡 **검색 팁:** 여러 단어를 검색할 때는 대문자로 `AND` 또는 `OR`을 사용해 보세요. (예: `APPLE AND AI`, `규제 OR 법안`)")
search_input = st.sidebar.text_input("🔎 검색어 입력", key="search_query").strip()

# 💡 [수정] 1차 필터링 (카테고리 다중 선택 로직)
# 아무것도 선택하지 않았을 때(리스트가 비어있을 때)는 전체 데이터를 보여주고,
# 하나라도 선택했다면 해당 카테고리들에 포함된(.isin) 데이터만 보여줍니다.
if not selected_categories:
    cat_df = df
else:
    cat_df = df[df['카테고리'].isin(selected_categories)]

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
# 6. 다중 정렬 및 최종 결과 렌더링
# ---------------------------------------------------------
# 필터링된 데이터프레임의 복사본을 만들어 정렬 작업 수행 (경고 메시지 방지)
sorted_df = filtered_df.copy()

# 1) 카테고리 사용자 정의 정렬 순서 지정 (여기에 없는 카테고리는 자동으로 맨 뒤로 밀림)
custom_order = ['기업 및 시장 동향', '기술 동향', '정책 및 규제 동향', '사회적 파급 효과', '기타']
sorted_df['카테고리'] = pd.Categorical(sorted_df['카테고리'], categories=custom_order, ordered=True)

# 2) 정렬 기준 및 방향(오름차순/내림차순) 세팅
# 기본 정렬: 1순위 도출_기간(내림차순), 2순위 카테고리(오름차순: 지정해둔 custom_order 순서)
sort_columns = ['도출_기간', '카테고리']
sort_ascending = [False, True] 

# 3) '기사_수' 컬럼 존재 여부에 따라 3순위 조건 동적 추가
if '기사_수' in sorted_df.columns:
    sort_columns.append('기사_수')
    sort_ascending.append(False) # 많은 순(내림차순)

# 4) 마지막 순위: 제목 가나다순 추가
sort_columns.append('제목')
sort_ascending.append(True)      # 가나다순(오름차순)

# 5) 설정한 조건으로 최종 정렬 실행
sorted_df = sorted_df.sort_values(by=sort_columns, ascending=sort_ascending)

# 6) 화면 렌더링
for index, row in sorted_df.iterrows():
    # 제목 설정 (도출_기간 포함)
    expander_title = f"[{row['도출_기간']}] 📌 {row['제목']} (키워드: {row['키워드']})"
    
    # 만약 기사 수 데이터가 있다면 제목에 배지처럼 표시 (선택 사항)
    if '기사_수' in row and pd.notna(row['기사_수']):
        expander_title += f" 🔥기사 {int(row['기사_수'])}건"
    
    with st.expander(expander_title):
        st.markdown("#### 📝 핵심 인사이트 요약")
        st.markdown(row['summary'])
        
        st.markdown(f"**📂 카테고리:** {row['카테고리']}")
        
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