"""10주차: PDF RAG 기반 Manager Multi-Agent (Streamlit)."""
import os
import time
import uuid     #식별자 생성 라이브러리. 사용자별 대화 세션 구분
import streamlit as st
from agents import Agent, Runner, FileSearchTool, SQLiteSession, set_default_openai_key

st.set_page_config(page_title="Week 10 Multi-Agent", page_icon="🧪")
st.title("10주차 | Multi-Agent Science Inquiry Coach")

try:
    set_default_openai_key(st.secrets["OPENAI_API_KEY"])
    vector_store_id = st.secrets["VECTOR_STORE_ID"]
except (KeyError, FileNotFoundError) as exc:
    st.error(f".streamlit/secrets.toml 설정을 확인하세요: {exc}")
    st.stop()

MODEL = st.secrets.get("MODEL", "gpt-4.1-mini")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("실습 설정")
    st.caption("6주차 PDF Vector Store 재사용")
    if st.button("새 대화 시작"):
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

rag = FileSearchTool(vector_store_ids=[vector_store_id], max_num_results=3)
inquiry_agent = Agent(
    name="Inquiry Coach",
    model=MODEL,
    instructions=(
        "중학생 과학탐구 질문을 개선하는 전문가다. 기존 PDF Knowledge를 검색하여 "
        "탐구 가능성, 변인, 측정 방법, 학년 적합성을 검토한다. "
        "학생 대신 정답이나 보고서를 완성하지 말고 질문과 개선 방향을 제시한다. "
        "근거가 없는 PDF 파일명이나 페이지 번호를 만들어내지 않는다."
    ),
    tools=[rag],
)
safety_agent = Agent(
    name="Safety Reviewer",
    model=MODEL,
    instructions=(
        "중학생 과학 실험 안전 검토 전문가다. 실험의 위험 요소와 교사 감독 필요성을 평가한다. "
        "유해 화학물질 혼합, 폭발, 화재 등 위험한 실험의 실행 절차는 제공하지 않는다. "
        "안전한 대체 활동을 제안한다."
    ),
)
manager = Agent(
    name="Science Inquiry Manager",
    model=MODEL,
    instructions=(
        "학생의 과학탐구를 돕는 교육용 Manager다. 탐구 질문의 타당성·변인·PDF 기준이 "
        "필요하면 review_inquiry를 호출하고, 안전 문제가 있거나 안전성 검토 요청이 있으면 "
        "check_safety를 호출한다. 복합 요청이면 두 전문가를 모두 활용한다. "
        "학생 수준의 친절한 한국어로 통합 피드백을 제공한다. "
        "전문가가 확인하지 않은 내용을 확인했다고 주장하지 않는다."
    ),
    tools=[
        inquiry_agent.as_tool(tool_name="review_inquiry", tool_description="PDF Knowledge 기반 과학탐구 질문 검토"),
        safety_agent.as_tool(tool_name="check_safety", tool_description="과학 실험의 안전성 및 교사 지도 필요성 검토"),
    ],
)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("과학탐구 질문을 입력하세요")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("전문 Agent 검토 중..."):
            try:
                # 동일한 세션 ID/DB 파일로 대화 이력 유지. 객체 자체는 재사용하지 않음.
                session = SQLiteSession(st.session_state.user_id, "week10_history.db")
                started = time.perf_counter()
                result = Runner.run_sync(manager, question, session=session)
                elapsed = time.perf_counter() - started
                answer = str(result.final_output)
                st.markdown(answer)
                st.caption(f"응답 시간: {elapsed:.1f}초")
                with st.expander("Manager 도구 호출 로그"):
                    for item in result.new_items:
                        if item.type == "tool_call_item":
                            raw = item.raw_item
                            st.code(f"{type(raw).__name__}: {getattr(raw, 'name', '')} {getattr(raw, 'arguments', '')}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"실행 오류: {type(exc).__name__}: {exc}")
