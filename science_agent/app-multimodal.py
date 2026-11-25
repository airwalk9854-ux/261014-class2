"""11주차: 멀티턴 대화 + 사이드바 이미지 업로드 + PDF RAG."""
import base64
import uuid
from pathlib import Path

import streamlit as st
from agents import Agent, FileSearchTool, Runner, SQLiteSession, set_default_openai_key

st.set_page_config(page_title="Multimodal Science Inquiry Coach", page_icon="🧪", layout="wide")
st.title("11주차 | Multimodal Science Inquiry Coach")
st.caption("멀티턴 대화 · 선택적 사진 첨부 · PDF Knowledge 기반 피드백")

try:
    set_default_openai_key(st.secrets["OPENAI_API_KEY"])
    vector_store_id = st.secrets["VECTOR_STORE_ID"]
except (KeyError, FileNotFoundError) as exc:
    st.error(f".streamlit/secrets.toml 설정을 확인하세요: {exc}")
    st.stop()

MODEL = st.secrets.get("MODEL", "gpt-4.1-mini")
DB_PATH = str(Path(__file__).with_name("week11_history.db"))

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0
if "agent_session" not in st.session_state:
    st.session_state.agent_session = SQLiteSession(st.session_state.session_id, DB_PATH)

rag = FileSearchTool(vector_store_ids=[vector_store_id], max_num_results=3)
agent = Agent(
    name="Multimodal Science Inquiry Coach",
    model=MODEL,
    instructions=(
        "너는 중학생 과학탐구 질문 형성을 돕는 교사형 AI다. "
        "이전 대화에서 다룬 실험, 이미지 분석 결과와 학생 질문의 맥락을 이어서 답한다. "
        "이미지에서 직접 관찰 가능한 사실과 추론을 명확히 구분하고, "
        "보이지 않는 정보는 추측하지 않는다. "
        "탐구 질문의 평가 기준이 필요하면 FileSearchTool로 PDF Knowledge를 검색한다. "
        "독립변인, 종속변인, 통제변인과 측정 가능성을 고려하여 피드백한다. "
        "학생 대신 과제를 완성하지 않고 개선 방향을 안내한다. "
        "위험한 실험은 교사 지도와 안전한 대안을 안내한다. "
        "확인되지 않은 PDF 페이지 번호나 이미지 세부사항을 만들어내지 않는다. "
        "사용자가 새 이미지를 첨부하지 않은 후속 질문에서도 이전 대화 맥락을 활용하되, "
        "원본 이미지를 다시 확인해야 할 경우 재첨부를 요청한다."
    ),
    tools=[rag],
)

with st.sidebar:
    st.header("실습 설정")
    st.subheader("사진 업로드")
    uploaded = st.file_uploader(
        "실험 사진 또는 탐구 활동지 (개인정보 없는 이미지)",
        type=["png", "jpg", "jpeg"],
        key=f"photo_{st.session_state.uploader_version}",
    )
    attach_image = st.checkbox(
        "다음 질문에 이 사진 첨부",
        value=True,
        disabled=uploaded is None,
        help="사진을 첨부하지 않아도 이전 대화에 대한 후속 질문은 가능합니다.",
    )
    if uploaded is not None:
        st.image(uploaded, caption="선택한 이미지", use_container_width=True)
        st.caption(f"파일 크기: {uploaded.size / (1024 * 1024):.2f} MiB")
    st.divider()
    st.caption("기존 PDF Vector Store를 재사용합니다.")
    if st.button("새 대화 시작", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.agent_session = SQLiteSession(st.session_state.session_id, DB_PATH)
        st.session_state.messages = []
        st.session_state.uploader_version += 1
        st.rerun()

if not st.session_state.messages:
    st.info("왼쪽에서 사진을 선택하거나, 아래 입력창에 과학탐구 질문을 입력하세요.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_bytes") is not None:
            st.image(message["image_bytes"], caption="질문에 첨부한 사진", width=320)
        st.markdown(message["content"])

question = st.chat_input("과학탐구에 대해 질문하거나 이전 답변에 이어 질문하세요.")
if question:
    question = question.strip()
    if not question:
        st.stop()

    image_bytes = None
    data_url = None
    if uploaded is not None and attach_image:
        image_bytes = uploaded.getvalue()
        if len(image_bytes) > 5 * 1024 * 1024:
            st.error("실습용 이미지는 5 MiB 이하로 제한합니다.")
            st.stop()
        mime = uploaded.type
        if mime not in ("image/png", "image/jpeg"):
            st.error("PNG 또는 JPEG 이미지만 사용할 수 있습니다.")
            st.stop()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{encoded}"

    content = [{"type": "input_text", "text": question}]
    if data_url is not None:
        content.append({"type": "input_image", "image_url": data_url})

    st.session_state.messages.append({
        "role": "user", "content": question, "image_bytes": image_bytes
    })
    with st.chat_message("user"):
        if image_bytes is not None:
            st.image(image_bytes, caption="질문에 첨부한 사진", width=320)
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                result = Runner.run_sync(
                    agent,
                    input=[{"role": "user", "content": content}],
                    session=st.session_state.agent_session,
                )
                answer = str(result.final_output)
                st.markdown(answer)
                with st.expander("도구 호출 확인"):
                    calls = [x for x in result.new_items if x.type == "tool_call_item"]
                    if not calls:
                        st.caption("이번 답변에서 기록된 도구 호출이 없습니다.")
                    for item in calls:
                        raw = item.raw_item
                        st.code(f"{type(raw).__name__}: {getattr(raw, 'name', '')}")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"실행 오류: {type(exc).__name__}: {exc}")
                st.caption("오류 후 대화 상태가 불확실할 수 있습니다. 필요하면 새 대화를 시작하세요.")
