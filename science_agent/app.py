import streamlit as st

from agents import Agent, Runner, SQLiteSession


# =========================================================
# 1. Page
# =========================================================

st.set_page_config(
    page_title="Science Inquiry Coach",
    page_icon="🔬"
)


# =========================================================
# 2. Agent
# =========================================================

science_agent = Agent(
    name="Science Inquiry Coach",

    instructions="""
    [ROLE]
    너는 중학생의 과학 자유탐구 질문 형성을 지원하는
    Science Inquiry Coach이다.

    [GOAL]
    학생이 자신의 관심사를 과학 개념과 연결하여
    탐구 가능한 질문을 스스로 발전시키도록 지원한다.

    [RULES]
    1. 학생의 생각을 먼저 확인한다.
    2. 완성된 탐구질문을 처음부터 바로 제공하지 않는다.
    3. 정보가 부족하면 추가 질문을 한다.
    4. 질문 개선 이유를 설명한다.
    5. 최종 선택은 학생이 한다.
    6. 수행평가나 탐구보고서를 대신 완성하지 않는다.
    7. 위험한 실험은 추천하지 않는다.

    [PROCESS]
    1. 학년과 관심분야를 확인한다.
    2. 초기 아이디어를 확인한다.
    3. 탐구 가능성을 분석한다.
    4. 개선할 부분을 설명한다.
    5. 개선 방향을 제안한다.
    6. 학생이 선택하거나 수정하도록 한다.
    7. 수정 결과를 다시 평가한다.

    [OUTPUT]
    상황에 따라 다음을 제공한다.
    - 현재 아이디어 분석
    - 개선이 필요한 부분
    - 개선 방향
    - 필요한 경우 질문 후보
    - 학생이 다음으로 생각할 질문
    """
)


# =========================================================
# 3. UI State
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# 4. Agent Session
# =========================================================

if "agent_session" not in st.session_state:

    st.session_state.agent_session = SQLiteSession(
        "student_session",
        "science_agent.db"
    )


# =========================================================
# 5. UI
# =========================================================

st.title("Science Inquiry Coach")

st.caption(
    "학생이 스스로 좋은 과학 탐구질문을 "
    "발전시키도록 지원하는 AI Agent"
)


# =========================================================
# 6. Previous Messages
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# 7. User Input
# =========================================================

user_input = st.chat_input(
    "과학 탐구에 관심 있는 주제를 입력하세요."
)


# =========================================================
# 8. Run Agent
# =========================================================

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):

        with st.spinner("생각하고 있습니다..."):

            try:

                result = Runner.run_sync(
                    science_agent,
                    user_input,
                    session=st.session_state.agent_session
                )

                response = result.final_output

                st.markdown(response)

            except Exception as e:

                response = f"오류가 발생했습니다: {e}"

                st.error(response)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
