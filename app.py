import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import os
import pickle

# ---------------------------------------------------------
# UI 스타일 강제 수정 (Multiselect 글자 잘림 방지)
# ---------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     /* multiselect 안의 태그(박스) 최대 너비를 100%로 늘림 */
#     .stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"] {
#         max-width: 150% !important;
#     }
    
#     /* 태그 안의 텍스트가 줄바꿈되거나 다 보이도록 말줄임표(ellipsis) 제거 */
#     .stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"] span {
#         white-space: normal !important;
#         overflow: visible !important;
#         text-overflow: unset !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

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
# 2. 데이터 및 임베딩 사전 로드 (다중 기간 병합 기능 포함)
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
    
    # 💡 [핵심] 최신 파일이 위로 올라오도록 내림차순 정렬 (reverse=True)
    return dict(sorted(file_map.items(), reverse=True))

@st.cache_data
def load_and_concat_data(selected_files_dict):
    """여러 기간의 Parquet 파일을 읽어와 하나로 합치는 함수"""
    df_list = []
    
    for period_label, file_path in selected_files_dict.items():
        temp_df = pd.read_parquet(file_path)
        start_date, end_date = period_label.split(" ~ ")
        short_date_format = f"{start_date[2:]}~{end_date[2:]}"
        temp_df['도출_기간'] = short_date_format
        df_list.append(temp_df)
        
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_embedding_dict(data_dir):
    dict_path = os.path.join(data_dir, 'keyword_embeddings_dict.pkl')
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            return pickle.load(f)
    return {}

# 1) 파일 목록 가져오기
file_map = get_available_periods(DATA_DIR)
if not file_map:
    st.error(f"'{DATA_DIR}' 폴더에서 분석 가능한 Parquet 파일을 찾을 수 없습니다.")
    st.stop()

# 2) 기간 리스트 추출 (최신순)
periods = list(file_map.keys())
latest_period = periods[0] # 가장 첫 번째 요소 = 가장 최신 파일

# 3) 사이드바: 카테고리처럼 다중 선택하는 기간 UI
st.sidebar.markdown("### 📅 분석 기간 설정")
st.sidebar.caption("보고 싶은 주차를 추가로 선택하여 병합할 수 있습니다.")

# 💡 [핵심 UI] default 값으로 가장 최신 주차를 넣어줍니다.
selected_periods = st.sidebar.multiselect(
    "조회할 기간 선택",
    options=periods,
    default=[latest_period], 
    placeholder="기간을 선택해주세요",
    format_func=lambda x: f"{x.split(' ~ ')[0][2:]}~{x.split(' ~ ')[1][2:]}" # 2026-03-30 ~ 2026-04-12 -> 26-03-30~26-04-12
)

# 4) 예외 처리: 사용자가 'X'를 눌러서 모든 기간을 지워버렸을 때
if not selected_periods:
    st.warning("⚠️ 최소 1개 이상의 기간을 선택해야 데이터를 볼 수 있습니다.")
    st.stop()

# 5) 선택된 기간에 해당하는 파일 경로만 딕셔너리로 추출
selected_files = {p: file_map[p] for p in selected_periods}

# 6) 필터링된 파일들을 병합하여 로드
df = load_and_concat_data(selected_files)
emb_dict = load_embedding_dict(DATA_DIR)

# 7) 메인 화면 상단 안내 문구
# 선택한 주차들을 화면에 텍스트로 가볍게 뿌려줍니다.
selected_periods_str = ", ".join([p.split(" ~ ")[0][2:] + "~" + p.split(" ~ ")[1][2:] for p in selected_periods])
st.caption(f"🕒 현재 조회 중인 기간: **{selected_periods_str}** (총 **{len(selected_files)}**개 주차 병합됨)")

# ---------------------------------------------------------
# 3. 사이드바: 필터 및 고급 검색창
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 및 고급 검색")

# 카테고리 리스트 추출
category_list = ['기업 및 시장 동향', '기술 동향', '정책 및 규제 동향', '사회적 파급 효과', '기타']

selected_categories = st.sidebar.multiselect(
    "📂 카테고리 선택", 
    options=category_list,
    default=[], 
    placeholder="전체보기 (클릭하여 카테고리 추가)"
)

# 고급 검색 안내 문구 추가
st.sidebar.caption("💡 **검색 팁:** 여러 단어를 검색할 때는 대문자로 `AND` 또는 `OR`을 사용해 보세요. (예: `APPLE AND AI`, `규제 OR 법안`)")
search_input = st.sidebar.text_input("🔎 검색어 입력", key="search_query").strip()

# 💡 [추가] 기사 수 정렬 라디오 버튼 추가
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 정렬 옵션")
sort_article_option = st.sidebar.radio(
    "🔥 토픽 내 기사 수 정렬",
    options=["기사 많은 순 (내림차순)", "기사 적은 순 (오름차순)"],
    index=0 # 기본값을 '기사 많은 순'으로 설정
)

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
sorted_df = filtered_df.copy()

# 1) 카테고리 사용자 정의 정렬 순서 지정
custom_order = ['기업 및 시장 동향', '기술 동향', '정책 및 규제 동향', '사회적 파급 효과', '기타']
sorted_df['카테고리'] = pd.Categorical(sorted_df['카테고리'], categories=custom_order, ordered=True)

# 💡 [필수 조치] '기사_수'가 문자열일 경우를 대비해 확실하게 숫자로 변환 (결측치는 0으로 처리)
if '기사_수' in sorted_df.columns:
    sorted_df['기사_수'] = pd.to_numeric(sorted_df['기사_수'], errors='coerce').fillna(0)

# 2) 정렬 기준 및 방향 동적 세팅
# 1순위는 무조건 최신 '도출_기간'
sort_columns = ['도출_기간']
sort_ascending = [False] 

# 💡 [핵심 수정] 사용자가 '기사 많은 순'을 골랐다면, 기사 수를 2순위(카테고리보다 앞)로 올립니다.
if '기사_수' in sorted_df.columns:
    if sort_article_option == "기사 많은 순 (내림차순)":
        sort_columns.extend(['기사_수', '카테고리']) # 기사_수가 앞
        sort_ascending.extend([False, True])
    else:
        # "기사 적은 순"이거나 기본값일 때는, 카테고리별로 예쁘게 묶어보는 게 더 중요할 수 있습니다.
        # 원하신다면 여기도 ['기사_수', '카테고리']로 바꾸셔도 됩니다.
        sort_columns.extend(['기사_수', '카테고리']) 
        sort_ascending.extend([True, True])
else:
    # 기사 수 데이터가 아예 없을 때의 기본 동작
    sort_columns.append('카테고리')
    sort_ascending.append(True)

# 3) 마지막 순위: 제목 가나다순
sort_columns.append('제목')
sort_ascending.append(True)

# 4) 최종 정렬 실행
sorted_df = sorted_df.sort_values(by=sort_columns, ascending=sort_ascending)

# 6) 화면 렌더링
for index, row in sorted_df.iterrows():
    # 제목 설정 (도출_기간 포함)
    expander_title = f"[{row['도출_기간']}] 📌 {row['제목']} (키워드: {row['키워드']})"
        
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
