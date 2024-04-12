import os
import streamlit as st
from datetime import datetime
from typing import Type
from pydantic import BaseModel
from pydantic import Field
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema.runnable import RunnablePassthrough
import nest_asyncio

nest_asyncio.apply()

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

st.title("D33 | OpenAI Assistants (졸업 과제)")
with st.expander("과제 내용 보기", expanded=False):
    # st.snow()
    st.markdown(
        """
    ### D31 (2024-04-10) 과제
    - 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
    - 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
    - 유저가 자체 OpenAI API 키를 사용하도록 허용하고,`st.sidebar` 내부의 `st.input`에서 이를 로드합니다.
    - `st.sidebar`를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.
    
    """
    )

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        # print(st.session_state["api_key"])

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

    st.divider()
    st.markdown(
        """
        GitHub 링크: https://github.com/LifeFi/py_w08_fullstack_gpt_d15/blob/d31_agents/d31_agents.ipynb
        """
    )

if "query" not in st.session_state:
    st.session_state["query"] = ""

if "result" not in st.session_state:
    st.session_state["result"] = ""


llm = ChatOpenAI(
    temperature=0.1,
    api_key=api_key if api_key else "_",
)


class WikipediaSearchTool(BaseTool):

    name = "WikipediaSearchTool"
    description = """
    Use this tool to find the website for the given query.
    """

    class WikipediaSearchToolArgsSchema(BaseModel):
        query: str = Field(
            description="The query you will search for. Example query: Research about the XZ backdoor",
        )

    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query):
        w = WikipediaAPIWrapper()
        return w.run(query)


class DuckDuckGoSearchTool(BaseTool):

    name = "DuckDuckGoTool"
    description = """
    Use this tool to find the website for the given query.
    """

    class DuckDuckGoSearchToolArgsSchema(BaseModel):
        query: str = Field(
            description="The query you will search for. Example query: Research about the XZ backdoor",
        )

    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)


class LoadWebsiteTool(BaseTool):

    name = "LoadWebsiteTool"
    description = """
    Use this tool to load the website for the given url.
    """

    class LoadWebsiteToolArgsSchema(BaseModel):
        url: str = Field(
            description="The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
        )

    args_schema: Type[LoadWebsiteToolArgsSchema] = LoadWebsiteToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        # transformer = Html2TextTransformer.transform_documents(docs)
        # print(docs)
        # with open("./outputs/research.txt", "w") as f:
        #     f.write(docs.page_content)
        return docs


class SaveToFileTool(BaseTool):
    name = "SaveToFileTool"
    description = """
    Use this tool to save the text to a file.
    """

    class SaveToFileToolArgsSchema(BaseModel):
        text: str = Field(
            description="The text you will save to a file.",
        )
        file_path: str = Field(
            description="Path of the file to save the text to.",
        )

    args_schema: Type[SaveToFileToolArgsSchema] = SaveToFileToolArgsSchema

    def _run(self, text, file_path):
        research_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./outputs", exist_ok=True)
        file_name = f"./outputs/{research_dt}_{file_path}"

        with open(file_name, "w", encoding="utf-8") as f:
            f.write(text)

        file_bytes = open(file_name, "rb").read()
        st.download_button(
            label="Download file",
            data=file_bytes,
            file_name=file_name,
            mime="text/plain",
        )

        return f"Text saved to {research_dt}_{file_path}"


def agent_invoke(input):

    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        tools=[
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
            LoadWebsiteTool(),
            SaveToFileTool(),
        ],
    )

    prompt = PromptTemplate.from_template(
        """    
        1. query 에 대해서 검색하고
        2. 검색 결과 목록에 website url 목록이 있으면, 각각의 website 내용을 text로 추출해서
        3. txt 파일로 저장해줘.
        4. 반드시 txt 내용도 모두 보여줘

        query: {query}    
        """,
    )

    chain = {"query": RunnablePassthrough()} | prompt | agent
    result = chain.invoke(input)
    return result["output"]


# query = "Research about the XZ backdoor"

# agent_invoke(query)


if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if api_key:
    st.subheader("What do you want reaearch about?")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "What do you want reaearch about?",
            key="query_input",
            value="Research about the XZ backdoor",
            label_visibility="collapsed",
        )
    with col2:
        run_agent = st.button(
            "Run Agent",
            key="run_button",
            type="primary",
            use_container_width=True,
        )


# END LOG: script run/rerun
end_time = datetime.now()
elapsed_time = end_time - start_time
elapsed_seconds = elapsed_time.total_seconds()
print(
    f"\n\033[43mEND Exec[{st.session_state['run_count']}]: {elapsed_seconds}s / {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
)
