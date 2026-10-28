"""URL에서 수집한 Knowledge를 검색하는 Streamlit Education Agent."""
import streamlit as st
from agents import Agent, FileSearchTool, Runner, SQLiteSession

st.set_page_config(page_title="Web Knowledge Education Agent", page_icon="📚")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
    vector_store_id = st.secrets["VECTOR_STORE_ID"]
except (KeyError, FileNotFoundError):
    st.error(".streamlit/secrets.toml에 OPENAI_API_KEY와 VECTOR_STORE_ID를 설정하세요.")
    st.stop()

# Agents SDK는 기본적으로 OPENAI_API_KEY 환경변수를 읽습니다.
from agents import set_default_openai_key
set_default_openai_key(api_key)

INSTRUCTIONS = """
[ROLE] 너는 교육 관련 웹 자료를 바탕으로 질문에 답하는 Education Agent이다.
[GOAL] 사용자가 신뢰할 수 있는 교육 자료의 내용을 이해하고 적용하도록 돕는다.
[KNOWLEDGE] 교육 정책, AI 역량, 교육 사례에 관한 사실 질문에는 FileSearchTool로
등록된 웹 Knowledge를 검색한 후 답한다.
[RULES]
1. 검색된 근거와 일반적인 제안을 구분한다.
2. 출처 문서 제목 또는 URL이 검색 결과에 있으면 함께 안내한다.
3. 자료에 없는 사실은 추측하지 않고 확인되지 않았다고 알린다.
4. 웹 Knowledge는 수집 당시의 내용이며 최신 내용과 다를 수 있음을 고려한다.
5. 학생의 학습이나 교사의 교육적 판단을 대신하지 않는다.
[OUTPUT] 핵심 답변, 근거, 교육적 활용 방안을 간결하게 제시한다.
"""

agent = Agent(
    name="Web Knowledge Education Agent",
    instructions=INSTRUCTIONS,
    tools=[FileSearchTool(vector_store_ids=[vector_store_id], max_num_results=5)],
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_session" not in st.session_state:
    st.session_state.agent_session = SQLiteSession(
        "web_education_student", "web_education_agent.db"
    )

st.title("Web Knowledge Education Agent")
st.caption("URL 수집 → Vector Store → FileSearchTool 기반 RAG 실습")
with st.sidebar:
    st.info("Knowledge를 변경하려면 setup_knowledge.py를 다시 실행하고 VECTOR_STORE_ID를 갱신하세요.")
    if st.button("새 대화 시작"):
        # 기존 대화와 분리된 세션 ID 사용
        import uuid
        st.session_state.agent_session = SQLiteSession(
            str(uuid.uuid4()), "web_education_agent.db"
        )
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("웹 Knowledge에 대해 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("자료를 검색하고 있습니다..."):
            try:
                result = Runner.run_sync(agent, prompt, session=st.session_state.agent_session)
                answer = str(result.final_output)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Agent 실행 오류: {exc}")
