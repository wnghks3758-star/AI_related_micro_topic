
import os
import numpy as np
import pandas as pd
import streamlit as st
import pickle

# 🚨 최신 LangChain 패키지 임포트
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import httpx
import warnings

# verify=False를 쓰면 터미널에 "보안상 위험할 수 있습니다"라는 
# 경고(InsecureRequestWarning)가 폭탄처럼 쏟아지므로, 이를 숨겨주는 코드입니다.
warnings.filterwarnings("ignore")

# ==========================================
# 1. 사이드바: 분석 대상 국가/지역 (체크박스 적용)
# ==========================================
st.sidebar.markdown("### 🌎 분석 대상 국가/지역")

# 지역별 베이스 디렉토리 매핑
region_map = {
    "미국+한국": "data_for_google_EN_KR",
    "중국": "data_for_google_HK_TW"
}

selected_regions = []

# 💡 체크박스로 UI 변경 (기본적으로 '미국+한국'은 체크된 상태로 설정)
if st.sidebar.checkbox("🇺🇸 미국 + 🇰🇷 한국", value=True):
    selected_regions.append("미국+한국")
    
if st.sidebar.checkbox("🇨🇳 중국 (홍콩/대만 포함)", value=False):
    selected_regions.append("중국")

if not selected_regions:
    st.warning("⚠️ 최소 1개 이상의 지역을 선택해야 합니다.")
    st.stop()

# 선택된 지역들의 CSV 데이터 폴더 경로 리스트 생성 (예: data_for_google_EN_KR/data)
data_dirs = [os.path.join(region_map[r], "articles") for r in selected_regions]

st.sidebar.markdown("---")

# ==========================================
# 2. 사이드바: 데이터 소스 설정
# ==========================================
st.sidebar.markdown("### 🗄️ 데이터 소스 설정")
data_source = st.sidebar.radio(
    "뉴스 수집 출처를 선택하세요",
    options=["Google News", "News API"],
    index=0
)
st.sidebar.markdown("---")

# ==========================================
# 3. 데이터 폴더 동시 스캔 함수
# ==========================================
@st.cache_data
def get_available_periods(data_dirs_list, source):
    """여러 지역의 data 폴더를 순회하며 공통/개별 주차 CSV 파일들을 묶어줍니다."""
    file_map = {}
    prefix = "NewsAPI_articles_AI_" if source == "News API" else "google_news_articles_AI_"

    for data_dir in data_dirs_list:
        if not os.path.exists(data_dir): 
            continue

        for file_name in os.listdir(data_dir):
            if source == "Google News" and "NewsAPI" in file_name:
                continue
                
            if file_name.startswith(prefix) and file_name.endswith(".csv"): # CSV 파일 기준
                date_str = file_name.replace(prefix, "").replace(".csv", "")
                try:
                    start_date, end_date = date_str.split("_to_")
                    label = f"{start_date} ~ {end_date}"
                    
                    # 💡 기간(label)을 키로, 해당 기간의 여러 지역 파일 경로를 리스트로 저장
                    if label not in file_map:
                        file_map[label] = []
                    file_map[label].append(os.path.join(data_dir, file_name))
                except ValueError:
                    pass
                    
    # 최신 기간이 위로 오도록 내림차순 정렬
    return dict(sorted(file_map.items(), reverse=True))

# 파일 목록 스캔
file_map = get_available_periods(data_dirs, data_source)

if not file_map:
    st.sidebar.error(f"선택하신 지역에서 '{data_source}' 기반의 CSV 파일을 찾을 수 없습니다.")
    st.stop()

# ==========================================
# 4. 사이드바: 분석 기간 설정
# ==========================================
st.sidebar.markdown("### 📅 분석 기간 설정")
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

selected_periods = chronological_periods[start_idx : end_idx + 1]

# 💡 4. 기존 데이터 병합 로직들이 '최신순' 기준으로 짜여 있으므로, 데이터 일관성을 위해 다시 최신순으로 뒤집어 줍니다.
selected_periods.reverse() 

if not selected_periods:
    st.warning("⚠️ 기간을 선택해야 데이터를 볼 수 있습니다.")
    st.stop()

# 3) 선택된 기간의 파일 경로 리스트 추출
selected_files = {p: file_map[p] for p in selected_periods}


# ==========================================
# 2. 벡터 DB 구축 (Streamlit 캐싱 적용하여 속도 최적화!)
# ==========================================
# 💡 @st.cache_resource: 질문을 던질 때마다 DB를 다시 만들지 않도록 메모리에 고정합니다.
@st.cache_resource(show_spinner=False)
def build_hybrid_retriever(_files_dict):
    texts = []
    metadatas = []
    vector_list = []
    
    for period_label, file_paths in _files_dict.items():
        
        # 💡 리스트 안에 있는 여러 지역의 파일 경로를 하나씩 꺼내도록 for문을 한 번 더 돕니다.
        for file_path in file_paths:
            
            # 1) CSV 로드
            try:
                temp_df = pd.read_csv(file_path, encoding='utf-8-sig').dropna(subset=['text']).reset_index(drop=True)
            except UnicodeDecodeError:
                temp_df = pd.read_csv(file_path, encoding='cp949').dropna(subset=['text']).reset_index(drop=True)

            # 2) NPY 로드 (이하 기존 코드와 동일)
            directory, file_name = os.path.split(file_path)
            new_directory = directory.replace("articles", "embeddings")

            # 3. 파일명 변경 (.csv -> .npy, articles_AI -> articles_AI_embedding)
            new_file_name = file_name.replace(".csv", ".npy").replace("articles_AI", "articles_embeddings_AI")

            # 4. OS 환경에 맞게(/ 또는 \) 다시 결합
            npy_path = os.path.join(new_directory, new_file_name)

            if not os.path.exists(npy_path):
                st.error(f"임베딩 파일을 찾을 수 없습니다: {npy_path}")
                continue
                
            temp_vectors = np.load(npy_path)

            # 3) 정합성 검증
            if len(temp_df) != len(temp_vectors):
                st.error(f"⚠️ 데이터 꼬임: {period_label}의 CSV 행({len(temp_df)})과 NPY 벡터({len(temp_vectors)}) 수가 다릅니다.")
                continue

            # 4) 데이터 매핑
            for i, row in temp_df.iterrows():
                content = str(row.get('contents_clean', row.get('text', ''))).strip()
                if not content: continue
                    
                texts.append(content)
                metadatas.append({
                    "title": row.get('title', row.get('제목', '제목 없음')),
                    "url": row.get('page_url', row.get('URL', '#')),
                    "date": period_label
                })
            vector_list.append(temp_vectors)

    if vector_list:
        all_vectors = np.vstack(vector_list)
        text_embedding_pairs = list(zip(texts, all_vectors.tolist()))
        
        # 💡 SSL 검사를 무시하는 커스텀 클라이언트 생성
        custom_http_client = httpx.Client(verify=False)
        
        # 💡 생성한 클라이언트를 임베딩 모델에 주입
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            http_client=custom_http_client,
            api_key=st.secrets['api_key']
        )
        
        # ------------------------------------------------------
        # 1. FAISS 벡터 검색기 (의미 기반) 구축
        # ------------------------------------------------------
        faiss_vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs, 
            embedding=embeddings_model, 
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
        )
        
        vector_retriever = faiss_vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.72, "k": 25} 
        )
        
        # ------------------------------------------------------
        # 2. 원본 문서를 LangChain Document 객체로 변환
        # ------------------------------------------------------
        # 💡 연구원님이 짚으신 바로 그 부분입니다! 텍스트와 메타데이터를 다시 묶어줍니다.
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        
        # ------------------------------------------------------
        # 3. BM25 키워드 검색기 (정확도 기반) 구축
        # ------------------------------------------------------
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 25
        
        # ------------------------------------------------------
        # 4. 하이브리드(앙상블) 검색기로 결합하여 반환
        # ------------------------------------------------------
        
        
        return vector_retriever,bm25_retriever
        
    return None,None

with st.spinner("선택된 기간의 데이터를 메모리에 로드하여 하이브리드 검색 DB를 구축하고 있습니다..."):
    # 이제 반환받는 객체는 vectorstore가 아니라 완벽하게 세팅된 retriever 자체입니다!
    vector_retriever,bm25_retriever = build_hybrid_retriever(selected_files)

if (vector_retriever is None) or (bm25_retriever is None):
    st.error("검색 DB를 구축할 수 없습니다. 데이터를 확인해 주세요.")
    st.stop()



# ==========================================
# 3. 최신 RAG 체인 구성 (LCEL 문법)
# ==========================================

# 🚨 반드시 {input} 변수를 사용해야 합니다!
prompt_template = """
당신은 글로벌 기술 동향, 산업 정책 및 경제 지표를 심층 분석하는 전문 수석 연구원입니다.
아래 제공된 [참고 문서]는 수집된 다국어 뉴스 기사 원문이며, 각 문서 앞에는 [문서 1], [문서 2]와 같이 고유 번호가 부여되어 있습니다.

[분석 및 작성 지침]
1. 다국어 교차 분석: 언어에 구애받지 말고 핵심 문맥과 사건을 정확히 종합하여 한국어로 요약하십시오.
2. 팩트 그라운딩: 철저하게 [참고 문서]에 명시된 사실에만 근거하십시오.
3. 📌 출처 주석 표기 (가장 중요): 작성하는 모든 문장의 끝에는 반드시 해당 사실의 근거가 된 문서 번호를 주석으로 달아주세요. (예: 삼성전자는 AI 메모리 수요 증가로 역대 최대 이익을 기록했다 [문서 1][문서 3].)
4. 학술적 보고서 톤 유지: "~함", "~임" 형태의 객관적인 문체를 사용하십시오.

[출력 구조]
- **핵심 요약**: 2~3줄 이내 요약 (주석 필수)
- **상세 분석**: 논리적 흐름에 따라 글머리 기호(-)를 활용하여 서술 (주석 필수)

[참고 문서]
{context}

[사용자 질문]
{input}

답변:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

custom_http_client = httpx.Client(verify=False)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=custom_http_client,
                 api_key=st.secrets['api_key'])


st.success("✅ RAG 시스템 준비 완료! 질문을 입력해 주세요.")

# ==========================================
# 4. Streamlit 채팅 UI 연동
# ==========================================

# 💡 [핵심] 기존의 복잡했던 체인 대신, 프롬프트와 LLM만 직접 연결합니다.
chain = PROMPT | llm

st.markdown("### 💬 AI 뉴스 분석 어시스턴트")
user_query = st.text_input("질문을 입력하세요 (예: 2026년 3월에 엔비디아와 삼성이 같이 한 일 요약해 줘)")



if user_query:
    with st.spinner("수집된 원문(의미+키워드)을 분석하여 답변을 생성 중입니다..."):

        # =========================================================
        # 1. 체크박스 선택값에 따른 번역 타겟 언어 동적 생성
        # =========================================================
        target_languages = [] # 기본적으로 한국어는 무조건 포함

        if "미국+한국" in selected_regions:
            target_languages.append("영어(English)")
            target_languages.appen("한국어")
            
        if "중국" in selected_regions:
            # 💡 홍콩/대만 뉴스 검색을 위해 '번체'를 명시적으로 지시합니다.
            target_languages.append("중국어 번체(Traditional Chinese, 繁體字)")
            
        # 예: "한국어, 영어(English), 중국어 번체(Traditional Chinese, 繁體字)"
        target_langs_str = ", ".join(target_languages)

        # =========================================================
        # 2. 동적 쿼리 확장 (핵심 명사 및 강조 키워드 추출 전용)
        # =========================================================
        expansion_prompt = PromptTemplate.from_template(
            "당신은 글로벌 뉴스 검색 엔진의 '핵심 키워드 추출 및 번역 전문가'입니다.\n"
            "사용자의 질문에서 문장 부호(따옴표 등)로 강조된 단어나, 검색에 필수적인 '명사(고유명사, 기술 용어, 기업명, 국가명 등)'만 엄격하게 추출하세요.\n"
            "🚨 [금지사항]: 서술어, 조사, 부사, 감탄사, 질문형 어미(예: ~알려줘, ~어때, 어떻게, 자세히 등)는 절대 포함하지 마십시오.\n"
            "추출된 핵심 키워드들을 반드시 '{target_langs}'로 각각 번역하여, 모두 공백으로 구분된 한 줄의 문자열로 출력하세요.\n\n"
            "[예시 1]\n"
            "검색어: 최근 \"마이크로소프트\"가 투자한 호주의 AI 교육에 대해 정리해줘\n"
            "출력 (미국+한국 선택 시): 마이크로소프트 Microsoft 호주 Australia AI 교육 Education\n\n"
            "[예시 2]\n"
            "검색어: 애플의 자율주행 전기차 프로젝트 취소 소식 알려줄래?\n"
            "출력 (중국 포함 시): 애플 Apple 蘋果 자율주행 Autonomous Driving 自動駕駛 전기차 EV 電動車 프로젝트 Project 專案 취소 Cancellation 取消\n\n"
            "검색어: {query}\n"
            "출력:"
        )

        # LLM 호출 (온도를 0으로 하여 창의성을 억제하고 기계적인 추출만 수행)
        llm_for_expansion = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=custom_http_client,
                 api_key=st.secrets['api_key'])

        response_msg = llm_for_expansion.invoke(
            expansion_prompt.format(target_langs=target_langs_str, query=user_query)
        )
        expanded_query = response_msg.content  # 텍스트만 쏙 빼내기
        
        # =========================================================
        # 1. 동적 쿼리 확장 (핵심 명사 추출) - 이전과 동일
        # =========================================================
        # (확장 프롬프트 실행하여 expanded_query 생성 완료 상태라고 가정)
        st.caption(f"🔍 **정밀 검색 키워드:** `{expanded_query}`")

        # =========================================================
        # 2. [투 트랙 검색] 엔진별로 가장 잘하는 쿼리 따로 던지기!
        # =========================================================
        # FAISS(벡터)에는 문맥이 살아있는 원본 질문을!
        faiss_docs = vector_retriever.invoke(user_query)
        
        # BM25(키워드)에는 번역된 알짜배기 명사들만!
        bm25_docs = bm25_retriever.invoke(expanded_query)

        # =========================================================
        # 3. 직접 RRF 융합 (Late Fusion) 및 중복 제거
        # =========================================================
        # 문서의 URL(또는 제목)을 고유 ID로 사용하여 중복을 제거하면서 점수를 합산합니다.
        rrf_scores = {}
        
        # 가중치 상수 (일반적으로 60을 많이 씁니다)
        k = 60 
        
        # FAISS 결과 점수 매기기 (1등은 1/61점, 2등은 1/62점...)
        for rank, doc in enumerate(faiss_docs):
            doc_id = doc.metadata.get("url", doc.page_content[:50]) # 고유 식별자
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (rank + 1 + k)

        # BM25 결과 점수 매기기
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("url", doc.page_content[:50])
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (rank + 1 + k)

        # 💡 점수가 높은 순으로(내림차순) 리스트 정렬
        fused_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # 최종적으로 점수는 떼어내고 문서 객체만 추출하여 상위 20개 컷!
        source_docs = [item["doc"] for item in fused_docs][:20]


        if not source_docs:
            st.warning("⚠️ 선택하신 기간의 데이터 중 질문과 관련도가 높은 기사를 찾지 못했습니다.")
            st.info("💡 **Tip:** 검색 기간을 더 넓게 선택하시거나, 다른 키워드로 질문해 보세요!")
            
        else:
            # 3. 문서 번호표 부착 및 텍스트 결합
            context_text = ""
            
            for i, doc in enumerate(source_docs):
                doc_number = i + 1
                doc.metadata["doc_index"] = doc_number
                # 💡 앙상블은 명시적 '점수' 대신 '순위(Rank)'를 보장하므로 점수 저장 로직은 삭제합니다.
                
                context_text += f"[문서 {doc_number}]\n제목: {doc.metadata.get('title')}\n내용: {doc.page_content}\n\n"
            
            # 4. LLM에 질문과 컨텍스트 전달하여 답변 생성
            result = chain.invoke({"context": context_text, "input": user_query})
            answer = result.content
            
            st.markdown("#### 🤖 AI 분석 결과")
            st.info(answer)
            
            # 5. 출처 출력 (점수 대신 융합 랭킹 순위로 표기)
            st.markdown("#### 📑 참고한 기사 출처 (하이브리드 랭킹순)")
            
            seen_urls = set()
            for doc in source_docs:
                url = doc.metadata.get("url", "")
                if url not in seen_urls:
                    idx = doc.metadata.get("doc_index")
                    title = doc.metadata.get("title", "제목 없음")
                    date = doc.metadata.get("date", "")
                    
                    # 점수 대신 [1], [2] 같은 랭킹 번호가 곧 중요도를 의미합니다.
                    st.markdown(f"**[{idx}]** [{date}] [{title}]({url})")
                    seen_urls.add(url)