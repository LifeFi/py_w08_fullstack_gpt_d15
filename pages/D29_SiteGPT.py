import re
from datetime import datetime
from langchain.document_loaders import SitemapLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import BaseOutputParser
from langchain.memory import ConversationBufferMemory
import streamlit as st

# START LOG: script run/rerun
if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0
st.session_state["run_count"] += 1

start_time = datetime.now()
print(
    f"\n\033[43mSTART Exec[{st.session_state['run_count']}]: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
)

st.set_page_config(
    page_title="SiteGPT | D29 과제",
    page_icon="📃",
)

st.title("D29 | SiteGPT")
with st.expander("과제 내용 보기", expanded=False):
    # st.snow()
    st.markdown(
        """
    ### D29 (2024-04-08) 과제
    - Cloudflare 공식문서를 위한 SiteGPT 버전을 만드세요.
    - 챗봇은 아래 프로덕트의 문서에 대한 질문에 답변할 수 있어야 합니다:
        - [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
        - [Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/)
        - [Workers AI](https://developers.cloudflare.com/workers-ai/)
    - [사이트맵](https://developers.cloudflare.com/sitemap.xml)을 사용하여 각 제품에 대한 공식문서를 찾아보세요.
    - 여러분이 제출한 내용은 다음 질문으로 테스트됩니다:
        - llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?
            - :blue[What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?]
        - Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?
            - :blue[What can I do with Cloudflare’s AI Gateway?]
        - 벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?
            - :blue[How many indexes can a single account have in Vectorize?]
    - 유저가 자체 OpenAI API 키를 사용하도록 허용하고, `st.sidebar` 내부의 `st.input`에서 이를 로드합니다.
    - `st.sidebar`를 사용하여 Streamlit app과 함께 깃허브 리포지토리에 링크를 넣습니다.
    
    """
    )

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        "**:blue[OpenAI API_KEY]**",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    url = st.text_input(
        "**:blue[Write down a URL]**",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
    )
    # 폴더 이름으로 사용.
    url_name = url.split("://")[1].replace("/", "_") if url else None

    st.divider()
    st.markdown(
        """
        GitHub 링크: https://github.com/LifeFi/py_w08_fullstack_gpt_d15/blob/d29_sitegpt/pages/D29_SiteGPT.py
        """
    )


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 최종 결과만 화면에 출력하고자 함.
# - ChatCallbackHandler 수정이 복잡하게 느껴져 llm 를 용도별로 분리.
llm_for_backstage = ChatOpenAI(
    temperature=0.1,
    api_key=api_key if api_key else "_",
)


llm = ChatOpenAI(
    temperature=0.1,
    api_key=api_key if api_key else "_",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


def parse_page(soup):
    # 대상 페이지와 무관한 내용이나 참고용으로 남겨둠.
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        # .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
            if "developers.cloudflare.com" in url_name
            else None
        ),
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    # print(docs)
    return docs


@st.cache_resource(show_spinner="Embedding docs...")
def embeded_docs(_docs, url_name):
    cache_dir = LocalFileStore(f"./.cache/sitegpt/embeddings/{url_name}")

    # https://platform.openai.com/docs/models/embeddings
    # https://platform.openai.com/account/limits
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(_docs, cached_embeddings)
    retriever = vector_store.as_retriever()

    return retriever


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                
    Examples:
                                                
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm_for_backstage
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


memory = ConversationBufferMemory(
    llm=llm_for_backstage,
    max_token_limit=1000,
    return_messages=True,
    memory_key="chat_history",
)


# [ 아직 미사용 ] => 이전 답변에서 유사한 질문 있는지 검색할때 사용할 예정
def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


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


if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if not url:
    st.warning("Please write down a **:blue[Sitemap URL]** on the sidebar.")


if api_key and url:
    if ".xml" not in url:
        st.warning("Please write down a Sitemap URL(**:blue[ .xml]**).")
    else:
        try:
            docs = load_website(url)
            docs_box = st.empty()
            retriever = embeded_docs(docs, url_name)
            send_message("I'm ready! Ask away!", "ai", save=False)
            paint_history()

            message = st.chat_input("Ask a question to the website.")
            if message:
                send_message(message, "human")

                chain = (
                    {
                        "docs": retriever,
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(get_answers)
                    | RunnableLambda(choose_answer)
                )

                def invoke_chain(question):
                    result = chain.invoke(question)
                    memory.save_context(
                        {"input": question},
                        {"output": result.content},
                    )
                    return result

                with st.chat_message("ai"):
                    invoke_chain(message)

        except Exception as e:
            print(e)
            e_str = str(e).lower()
            match = re.search(r"(api)(_|-|\s)(key)", e_str)
            if match:
                st.error("API_KEY 를 확인해 주세요.")

            st.expander("Error Details", expanded=True).write(f"Error: {e}")
            docs_box.write(docs)


# END LOG: script run/rerun
end_time = datetime.now()
elapsed_time = end_time - start_time
elapsed_seconds = elapsed_time.total_seconds()
print(
    f"\n\033[43mEND Exec[{st.session_state['run_count']}]: {elapsed_seconds}s / {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
)
