import streamlit as st
from datetime import datetime
import functools


def time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # START LOG: script run/rerun
        if "run_count" not in st.session_state:
            st.session_state["run_count"] = 0
        st.session_state["run_count"] += 1

        start_time = datetime.now()
        print(
            f"\n\033[43mSTART Exec[{st.session_state['run_count']}]: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
        )
        result = func(*args, **kwargs)
        # END LOG: script run/rerun
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        print(
            f"\n\033[43mEND Exec[{st.session_state['run_count']}]: {elapsed_seconds}s / {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')} ===============================\033[0m"
        )
        return result

    return wrapper
