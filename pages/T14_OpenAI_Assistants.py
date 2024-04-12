import asyncio
import os
import json
from datetime import datetime
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from openai import OpenAI
import yfinance
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


st.title("보너스 챕터14. OpenAI Assistants API")

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

if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]


@st.cache_data
def create_assistant():
    return client.beta.assistants.create(
        name="Investor Assistant",
        instructions="You help users do research on publicly traded companies and you help users decide if they should buy the stock or not.",
        model="gpt-4-1106-preview",
        tools=functions,
    )


if st.button("Recreate Assistant", type="primary"):
    create_assistant.clear()

with st.status("Creating Assistant...", expanded=True) as status:
    assistant = create_assistant()
    status.update(label="Assistant Created!", state="complete")

st.write(assistant.id)


message = st.text_input(
    label="Message",
    value="Now I want to know if Cloudflare is a good buy",
)


@st.cache_data
def create_thread():
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ]
    )


if st.button("Recreate Thread", type="primary"):
    create_thread.clear()

with st.status("Creating Thread...", expanded=True) as status:
    thread = create_thread()
    status.update(label="Thread Created!", state="complete")

st.write(thread.id)


@st.cache_data
def create_run(thread_id, assistant_id):
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


if st.button("Recreate Run", type="primary"):
    create_run.clear()


with st.status("Creating Run...", expanded=True) as status:
    run = create_run(thread.id, assistant.id)
    st.write(run.id)
    status.update(label="Run Created!", state="complete")

if st.button("Cancel Run"):
    client.beta.threads.runs.cancel(run_id=run.id, thread_id=thread.id)


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


if st.button("Get Run Status"):
    get_run_item = get_run(run.id, thread.id)
    st.write(get_run_item.id)
    st.write(get_run_item.assistant_id)
    st.write(get_run_item.status)
    st.write(get_run_item.completed_at)


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
        st.write(f"{message.role}: {message.content[0].text.value}")


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    # st.write(run.required_action)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        # st.write(action)
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


run_status_box = st.empty()


if st.button("Send Message", type="primary"):
    send_message(thread.id, message)

if st.button("Get Messages"):
    st.write(get_messages(thread.id))


if st.button("Get Tool Outputs"):
    result = get_tool_outputs(run.id, thread.id)


if st.button("Submit Tool Outputs", type="primary"):
    result = submit_tool_outputs(run.id, thread.id)


async def get_run_status(run_id, thread_id):
    with run_status_box.status("Getting Run Status...") as status:
        # status : [ "queued", "in_progress", "completed", "requires_action", "expired", "cancelling", "cancelled", "failed" ]
        waiting_state = ("queued", "in_progress", "cancelling")
        while True:
            run = get_run(run_id, thread_id)
            if run.status not in waiting_state:
                status.update(label=run.status, state="complete")
                break
            status.update(label=run.status, state="running")
            await asyncio.sleep(1)


asyncio.run(get_run_status(run.id, thread.id))
st.write(":red[asyncio.run() Done!]")

#
# with st.status(":red[Polling Run Status...]", expanded=True) as status:
#     run = client.beta.threads.runs.poll(
#         run.id,
#         thread.id,
#         poll_interval_ms=500,
#         timeout=10000,
#     )
#     status.update(label=run.status, state="complete")


get_messages(thread.id)


# END LOG: script run/rerun
end_time = datetime.now()
elapsed_time = end_time - start_time
elapsed_seconds = elapsed_time.total_seconds()
print(
    f"\n\033[43mEND Exec[{st.session_state['run_count']}]: {elapsed_seconds}s / {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
)
