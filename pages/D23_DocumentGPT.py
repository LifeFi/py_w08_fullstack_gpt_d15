from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

# Steramlit code
st.set_page_config(
    page_title="D23 | FullstackGPT 과제",
    page_icon="🌪️",
)
header = st.container()
header.title("D23 | FullstackGPT 과제")
with header.expander("과제 내용 보기"):
    st.snow()
    st.markdown(
        """
    ### D23 (2024-04-02) 과제
    - D19 과제를 Streamlit으로 만들고 배포하세요.

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

if header.button("RESET"):
    st.session_state["messages"] = []

""" header 고정을 위한 처리.

Streamlit 에서는 html 를 text로 처리하기 때문에, unsafe_allow_html=True 옵션 추가 필요.
[data-testid="stVerticalBlock"]
- strealit 이 elemnet 에 data-testid 를 부여한다고 함.
"""

header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
header.markdown(
    """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 20px;
            background-color: white;
            z-index: 999; 
        } 
        .fixed-header { 
            border-bottom: 1px solid rgba(0,0,0,0.2);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

cache_dir = LocalFileStore("./.cache/")
print(cache_dir.root_path)

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

loader = UnstructuredFileLoader("./files/document.txt")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(docs, cached_embeddings)
retriver = vectorstore.as_retriever()

memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    memory_key="chat_history",
)

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
    result = chain.invoke(question)
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result.content)
    return result


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


send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything about Chapter 3...")
if message:
    send_message(message, "human")

    with st.chat_message("ai"):
        invoke_chain(message)
