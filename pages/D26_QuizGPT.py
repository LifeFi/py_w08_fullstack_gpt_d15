import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

import streamlit as st
from datetime import datetime

start_time = datetime.now()
print(
    f"\n\033[43mSTART Exec: {start_time.strftime('%H:%M:%S.%f')} =========================================\033[0m"
)

st.set_page_config(
    page_title="QuizGPT | D26 과제",
    page_icon="☘️",
)

st.title("D26 | QuizGPT Turbo")
with st.expander("과제 내용 보기", expanded=False):
    # st.snow()
    st.markdown(
        """
    ### D63 (2024-04-05) 과제
    QuizGPT를 구현하되 다음 기능을 추가합니다:
    - 함수 호출을 사용합니다.
    - 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
    - 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
    - 만점이면 `st.ballons`를 사용합니다.
    - 유저가 자체 OpenAI API 키를 사용하도록 허용하고, `st.sidebar` 내부의 `st.input`에서 로드합니다.
    - `st.sidebar`를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
    """
    )

with st.sidebar:
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""

    if "quiz_subject" not in st.session_state:
        st.session_state["quiz_subject"] = ""

    api_key_input = st.empty()

    def reset_api_key():
        st.session_state["api_key"] = ""
        print(st.session_state["api_key"])

    if st.button(":red[Reset API_KEY]"):
        reset_api_key()

    api_key = api_key_input.text_input(
        ":blue[OpenAI API_KEY]",
        value=st.session_state["api_key"],
        key="api_key_input",
    )

    if api_key != st.session_state["api_key"]:
        st.session_state["api_key"] = api_key
        st.rerun()

    print(api_key)

    st.divider()
    st.markdown(
        """
        GitHub 링크: https://github.com/LifeFi/py_w08_fullstack_gpt_d15/blob/d26_quizgpt/pages/D26_QuizGPT.py
        """
    )

if not api_key:
    st.warning("Please provide an :blue[OpenAI API Key] on the sidebar.")

else:
    try:

        @st.cache_data(show_spinner="퀴즈를 맛있게 굽고 있어요...")
        def run_quiz_chain(*, subject, count, difficulty):
            chain = prompt | llm
            return chain.invoke(
                {
                    "subject": subject,
                    "count": count,
                    "difficulty": difficulty,
                }
            )

        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                """
            #### 자~ 이제 퀴즈를 만들어 볼까요?
            """
            )
        with col2:

            def reset_quiz():
                quiz_subject = ""
                st.session_state["quiz_subject"] = ""
                run_quiz_chain.clear()

            # 제대로 동작하지 않음. => 수정 필요
            if st.button(":red[퀴즈 초기화]"):
                reset_quiz()

        with st.form("quiz_create_form"):

            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                quiz_subject = st.text_input(
                    ":blue[주제]",
                    placeholder="무엇을 주제로 퀴즈를 만들까요?",
                    value=st.session_state["quiz_subject"],
                    # label_visibility="collapsed",
                )

            with col2:
                quiz_count = st.number_input(
                    ":blue[개수]",
                    placeholder="개수",
                    value=10,
                    min_value=2,
                    # label_visibility="collapsed",
                )

            with col3:
                quiz_difficulty = st.selectbox(
                    ":blue[레벨]",
                    ["1", "2", "3", "4", "5"],
                    # label_visibility="collapsed",
                )

            st.form_submit_button(":blue[퀴즈 만들기 시작]", use_container_width=True)

        function = {
            "name": "create_quiz",
            "description": "function that takes a list of questions and answers and returns a quiz",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                },
                                "answers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "answer": {
                                                "type": "string",
                                            },
                                            "correct": {
                                                "type": "boolean",
                                            },
                                        },
                                        "required": ["answer", "correct"],
                                    },
                                },
                            },
                            "required": ["question", "answers"],
                        },
                    }
                },
                "required": ["questions"],
            },
        }
        # ChatOpenAI model 정보
        # - https://platform.openai.com/docs/models/gpt-3-5-turbo
        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-3.5-turbo-0125",
            temperature=0.1,
        ).bind(
            function_call="auto",
            functions=[
                function,
            ],
        )

        prompt = PromptTemplate.from_template(
            """            
            Please create a quiz based on the following criteria:

            Topic: {subject}
            Number of Questions: {count}
            Difficulty Level: Level-{difficulty}/5
            Language: Korean

            The quiz should be well-structured with clear questions and correct answers.
            Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
            The quiz format should be multiple-choice,
            and each question should be accompanied by four possible answers, with only one correct option.
            """,
        )

        if quiz_subject:
            response_box = st.empty()
            response = run_quiz_chain(
                subject=quiz_subject,
                count=quiz_count,
                difficulty=quiz_difficulty,
            )
            response = response.additional_kwargs["function_call"]["arguments"]
            response = json.loads(response)

            generated_quiz_count = len(response["questions"])

            with st.form("quiz_questions_form"):
                solved_count = 0
                correct_count = 0
                for index, question in enumerate(response["questions"]):
                    st.write(f"{index+1}. {question['question']}")
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        label_visibility="collapsed",
                        key=f"question_{index}",
                    )
                    if value:
                        solved_count += 1
                        if {"answer": value, "correct": True} in question["answers"]:
                            st.success("정답! :100:")
                            correct_count += 1
                        else:
                            st.error("다시 도전해 보아요! :sparkles:")
                st.divider()
                # st.write(solved_count, correct_count, generated_quiz_count)

                result = st.empty()

                submitted = st.form_submit_button(
                    (
                        ":blue[제출하기]"
                        if solved_count < generated_quiz_count
                        else (
                            ":blue[:100: 축하합니다~ 새로운 주제로 도전해 보세요!]"
                            if correct_count == generated_quiz_count
                            else ":blue[다시 도전 💪]"
                        )
                    ),
                    use_container_width=True,
                    disabled=correct_count == generated_quiz_count,
                )
                if correct_count == generated_quiz_count:
                    for _ in range(3):
                        st.balloons()

                if solved_count == generated_quiz_count:
                    result.subheader(
                        f"결과: :blue[{correct_count}] / {generated_quiz_count}"
                    )
                print("submitted: ", submitted)

                if submitted:
                    print("submitted: ", submitted)

                    if solved_count < generated_quiz_count:
                        result.error(
                            f"퀴즈를 모두 풀고 제출해 주세요. ( 남은 퀴즈 개수: :red[{generated_quiz_count - solved_count}] / 답변한 퀴즈 개수: :blue[{solved_count}] )"
                        )

    except Exception as e:
        if (
            "api_key" in str(e)
            or "api-key" in str(e)
            or "API key" in str(e)
            or "API Key" in str(e)
        ):
            st.error("API_KEY 를 확인해 주세요.")
        st.expander("Error Details", expanded=True).write(f"Error: {e}")

        if "response" in locals():
            response_box.json(response)


end_time = datetime.now()
elapsed_time = end_time - start_time
elapsed_seconds = elapsed_time.total_seconds()
print(
    f"\n\033[43mEND Exec: {elapsed_seconds}s / {end_time.strftime('%H:%M:%S.%f')} =========================================\033[0m"
)
