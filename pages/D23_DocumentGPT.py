from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.schema.document import Document
import os

# from datetime import datetime
# import time

# Steramlit code
st.set_page_config(
    page_title="DocumentGPT | D23 과제",
    page_icon="🌪️",
)

st.write(os.getcwd())

with st.sidebar:
    # date = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    # st.header(date)
    file = st.file_uploader(
        ":blue[Upload a .txt .pdf or .docx file]",
        type=["pdf", "txt", "docx", "md"],
    )

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.text_input(
        ":blue[OpenAI API_KEY]",
        value=st.session_state["api_key"],
        key="api_key_input",
    )
    st.session_state["api_key"] = api_key

    # print(api_key)

    def reset_api_key():
        st.session_state["api_key"] = ""
        # print(st.session_state.api_key)

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()
        st.rerun()

    st.divider()
    st.markdown(
        """
        GitHub 링크: https://github.com/LifeFi/py_w08_fullstack_gpt_d15/blob/d23/pages/D23_DocumentGPT.py
    
        """
    )


header = st.container()
header.title("D23 | FullstackGPT 과제")
with header.expander("과제 내용 보기", expanded=False if file and api_key else True):
    # st.snow()
    st.markdown(
        """
    ### D23 (2024-04-02) 과제
    - D19 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
    - 파일 업로드 및 채팅 기록을 구현합니다.
    - 사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
    - st.sidebar에 스트림릿 앱의 코드가 담긴 깃허브 리포지토리 링크를 넣습니다.

    #### [참고] D19 과제
    -   Stuff Documents 체인을 사용하여 완전한 RAG 파이프라인을 구현하세요.
    -   체인을 수동으로 구현해야 합니다.
    -   체인에 `ConversationBufferMemory`를 부여합니다.
    -   이 문서를 사용하여 RAG를 수행하세요: [https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223](https://gist.github.com/serranoarevalo/5acf755c2b8d83f1707ef266b82ea223)
    -   체인에 다음 질문을 합니다:
        -   is Aaronson guilty?
        -   What message did he write in the table?
        -   Who is Julia?
    """
    )

if header.button(":red[RESET]"):
    st.session_state["messages"] = []

# header 고정을 위한 처리.
# Streamlit 에서는 html 를 text로 처리하기 때문에, unsafe_allow_html=True 옵션 추가 필요.
# [data-testid="stVerticalBlock"]
# - strealit 이 elemnet 에 data-testid 를 부여한다고 함.

header.write(
    """
    <div class='fixed-header'/>
    """,
    unsafe_allow_html=True,
)
header.markdown(
    """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 20px;
            z-index: 999; 
        } 
        .fixed-header { 
            border-bottom: 1px solid rgba(0,0,0,0.2);
        }
        @media (prefers-color-scheme: dark) {
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                background-color: black;
            }
        }
        @media (prefers-color-scheme: light) {
            div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
                background-color: white;
            }
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

if not file:
    st.warning("Please upload a :blue[File] on the sidebar.")

if not api_key:
    st.warning("Please provide an :blue[OpenAI API Key] on the sidebar.")


# 로직 code
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


try:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        openai_api_key=api_key,
    )
except Exception as e:
    if "api_key" or "api-key" or "API key" or "API Key" in str(e):
        st.error("ChatOpenAI() : API_KEY 를 확인해 주세요.")
    else:
        st.error(f"Error: {e}")
    st.stop()

# print(llm, "=======================================")
# print(llm.openai_api_key, "=======================================")
# print(api_key, "=======================================")


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    except Exception as e:
        if "api_key" or "api-key" or "API key" or "API Key" in str(e):
            st.error("OpenAIEmbeddings() : API_KEY 를 확인해 주세요.")
        else:
            st.error(f"Error: {e}")
        st.stop()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


try:
    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True,
        memory_key="chat_history",
    )
except Exception as e:
    if "api_key" or "api-key" or "API key" or "API Key" in str(e):
        st.error("ConversationBufferMemory() : API_KEY 를 확인해 주세요.")
    else:
        st.error(f"Error: {e}")
    st.stop()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant.
            Answer questions using only the following context.
            If you don't know the answer just say you don't know, don't make it up:
            \n\n{context}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


# streamlit code


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


if file and api_key:
    retriver = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about Chapter 3...")
    if message:
        send_message(message, "human")

        chain = (
            {
                "context": retriver,
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(load_memory),
            }
            | prompt
            | llm
        )

        def invoke_chain(question):
            # print(chain)

            try:
                result = chain.invoke(question)
            except Exception as e:
                if "api_key" or "api-key" or "API key" or "API Key" in str(e):
                    st.error("chain.invoke() : API_KEY 를 확인해 주세요.")
                else:
                    st.error(f"Error: {e}")
                st.stop()

            memory.save_context(
                {"input": question},
                {"output": result.content},
            )
            print(result.content)
            return result

        with st.chat_message("ai"):
            invoke_chain(message)
else:
    st.session_state["messages"] = []
