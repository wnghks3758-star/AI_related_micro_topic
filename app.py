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
# 2. 데이터 및 임베딩 사전 로드 (다중 지역 병합 완벽 지원)
# ---------------------------------------------------------

# 💡 [수정] 사이드바 최상단: 분석 대상 국가/지역 복수 선택 (Multiselect)
st.sidebar.markdown("### 🌎 분석 대상 국가/지역")
selected_regions = st.sidebar.multiselect(
    "분석할 지역을 모두 선택하세요",
    options=["미국+한국", "중국"],
    default=["미국+한국"], # 기본적으로 하나는 선택되어 있도록 설정
    placeholder="지역을 선택해주세요"
)

if not selected_regions:
    st.warning("⚠️ 최소 1개 이상의 지역을 선택해야 합니다.")
    st.stop()

# 💡 선택된 지역들에 대한 경로 매핑
region_map = {
    "미국+한국": "data_for_google_EN_KR",
    "중국": "data_for_google_HK_TW"
}

# 선택된 지역들의 데이터(result) 및 임베딩 경로 리스트 생성
result_dirs = [os.path.join(region_map[r], "result") for r in selected_regions]
emb_dirs = [os.path.join(region_map[r], "embeddings") for r in selected_regions]

st.sidebar.markdown("---")

# 데이터 소스 선택
st.sidebar.markdown("### 🗄️ 데이터 소스 설정")
data_source = st.sidebar.radio(
    "뉴스 수집 출처를 선택하세요",
    options=["Google News", "News API"],
    index=0
)

st.sidebar.markdown("---")

@st.cache_data
def get_available_periods(data_dirs_list, source):
    """여러 지역 폴더를 순회하며 공통/개별 주차 파일들을 묶어주는 함수"""
    file_map = {}
    prefix = "Micro_Topics_NewsAPI_" if source == "News API" else "Micro_Topics_"

    for data_dir in data_dirs_list:
        if not os.path.exists(data_dir): 
            continue
            
        for file_name in os.listdir(data_dir):
            if source == "Google News" and "NewsAPI" in file_name:
                continue

            if file_name.startswith(prefix) and file_name.endswith(".parquet"):
                date_str = file_name.replace(prefix, "").replace(".parquet", "")
                try:
                    start_date, end_date = date_str.split("_to_")
                    label = f"{start_date} ~ {end_date}"
                    
                    # 💡 기간(label)을 키로 삼고, 해당 기간의 여러 지역 파일 경로를 리스트로 저장
                    if label not in file_map:
                        file_map[label] = []
                    file_map[label].append(os.path.join(data_dir, file_name))
                except ValueError:
                    pass
    
    return dict(sorted(file_map.items(), reverse=True))

@st.cache_data
def load_and_concat_data(selected_files_dict):
    """선택된 주차에 속한 모든 지역의 파일을 하나로 합치는 함수"""
    df_list = []
    for period_label, file_paths in selected_files_dict.items():
        for file_path in file_paths:
            temp_df = pd.read_parquet(file_path)
            
            start_date, end_date = period_label.split(" ~ ")
            short_date_format = f"{start_date[2:]}～{end_date[2:]}"
            temp_df['도출_기간'] = short_date_format
            
            # 💡 [디테일 추가] 어느 지역에서 온 데이터인지 칼럼 추가
            if "EN_KR" in file_path:
                temp_df['데이터_지역'] = "🇺🇸미국+🇰🇷한국"
            elif "HK_TW" in file_path:
                temp_df['데이터_지역'] = "🇨🇳중국(HK_TW)"
                
            df_list.append(temp_df)
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def load_embedding_dict(emb_dirs_list, source):
    """여러 지역의 임베딩 사전을 읽어와 하나로 통합하는 함수"""
    dict_filename = 'keyword_embeddings_dict_NewsAPI.pkl' if source == "News API" else 'keyword_embeddings_dict.pkl'
    combined_dict = {}
    
    for emb_dir in emb_dirs_list:
        dict_path = os.path.join(emb_dir, dict_filename)
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                temp_dict = pickle.load(f)
                combined_dict.update(temp_dict) # 💡 파이썬 update()로 딕셔너리 안전하게 병합
                
    return combined_dict

# 1) 파일 목록 가져오기 (여러 지역 경로 전달)
file_map = get_available_periods(result_dirs, data_source)

if not file_map:
    st.error(f"⚠️ 선택하신 지역의 ({data_source}) 데이터를 찾을 수 없습니다.")
    st.stop()

# 2) 기간 리스트 추출 및 사이드바 선택 UI
periods = list(file_map.keys())
latest_period = periods[0]

st.sidebar.markdown("### 📅 분석 기간 설정")
st.sidebar.caption("양쪽 끝의 슬라이더를 움직여 조회할 기간을 설정하세요.")

# 💡 1. 슬라이더 UI의 직관성을 위해 기간 리스트를 '과거 ➡️ 최신' 순서로 뒤집습니다.
chronological_periods = list(reversed(periods))

# 💡 2. Multiselect 대신 Select Slider 사용 (범위 지정)
selected_range = st.sidebar.select_slider(
    "조회할 기간 범위 선택",
    options=chronological_periods,
    value=(latest_period, latest_period), # 기본값: 가장 최신 주차 1개만 선택된 상태
    format_func=lambda x: f"{x.split(' ~ ')[0][2:]}～{x.split(' ~ ')[1][2:]}"
)

# 💡 3. 슬라이더에서 선택된 시작점과 끝점(튜플)을 받아, 그사이에 있는 모든 주차를 리스트로 추출합니다.
start_period, end_period = selected_range
start_idx = chronological_periods.index(start_period)
end_idx = chronological_periods.index(end_period)

# 선택된 범위 안의 모든 기간을 가져옴
selected_periods = chronological_periods[start_idx : end_idx + 1]

# 💡 4. 기존 데이터 병합 로직들이 '최신순' 기준으로 짜여 있으므로, 데이터 일관성을 위해 다시 최신순으로 뒤집어 줍니다.
selected_periods.reverse() 

if not selected_periods:
    st.warning("⚠️ 기간을 선택해야 데이터를 볼 수 있습니다.")
    st.stop()

# 3) 선택된 기간의 파일 경로 리스트 추출
selected_files = {p: file_map[p] for p in selected_periods}

# 4) 병합된 데이터프레임 및 통합 임베딩 사전 로드
df = load_and_concat_data(selected_files)
emb_dict = load_embedding_dict(emb_dirs, data_source)

# 5) 메인 화면 상단 안내 문구
selected_periods_str = ", ".join([p.split(" ~ ")[0][2:] + "～" + p.split(" ~ ")[1][2:] for p in selected_periods])
regions_str = ", ".join(selected_regions)
st.caption(f"🌎 지역: **{regions_str}** | 📌 소스: **{data_source}** | 🕒 조회 기간: **{selected_periods_str}**")

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
    "🔥 정렬 방식",
    options=["관련도 순", "중요도 순"],
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
# 필터링된 데이터프레임의 복사본을 만들어 정렬 작업 수행 (경고 메시지 방지)
sorted_df = filtered_df.copy()

# 1) 카테고리 사용자 정의 정렬 순서 지정 (여기에 없는 카테고리는 자동으로 맨 뒤로 밀림)
custom_order = ['기업 및 시장 동향', '기술 동향', '정책 및 규제 동향', '사회적 파급 효과', '기타']
sorted_df['카테고리'] = pd.Categorical(sorted_df['카테고리'], categories=custom_order, ordered=True)

# 2) 정렬 기준 및 방향(오름차순/내림차순) 세팅
# 기본 정렬: 1순위 도출_기간(내림차순), 2순위 카테고리(오름차순: 지정해둔 custom_order 순서)
sort_columns = ['도출_기간', '카테고리']
sort_ascending = [False, True] 

# 3) 💡 [수정] 사용자가 선택한 정렬 옵션에 따라 점수 칼럼 동적 추가
if sort_article_option == "관련도 순":
    if 'relevance_score' in sorted_df.columns:
        sort_columns.append('relevance_score')
        sort_ascending.append(False) # 높은 점수부터(내림차순)
elif sort_article_option == "중요도 순":
    if 'importance_score' in sorted_df.columns:
        sort_columns.append('importance_score')
        sort_ascending.append(False) # 높은 점수부터(내림차순)

# 4) 마지막 순위: 제목 가나다순 추가
sort_columns.append('제목')
sort_ascending.append(True)      # 가나다순(오름차순)

# 5) 설정한 조건으로 최종 정렬 실행
sorted_df = sorted_df.sort_values(by=sort_columns, ascending=sort_ascending)

# 6) 화면 렌더링
for index, row in sorted_df.iterrows():
    # 제목 설정 (도출_기간 포함)
    expander_title = f"[{row['도출_기간'].replace('~','～')}] 📌 {row['제목']} (키워드: {row['키워드']})"
    
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