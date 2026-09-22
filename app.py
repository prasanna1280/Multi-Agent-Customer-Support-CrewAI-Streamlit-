import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from crewai.tools import tool


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# ============================================================
# 2. VALIDATE API KEYS
# ============================================================

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured in the .env file.")
    st.stop()

if not SERPER_API_KEY:
    st.error("SERPER_API_KEY is not configured in the .env file.")
    st.stop()


# Make sure CrewAI can see the keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY


# ============================================================
# 3. ENTRY AGENT FILE-WRITING TOOL
# ============================================================

@tool("save_answers_to_file")
def save_answers_to_file(
    query: str,
    answer_1: str,
    answer_2: str
) -> str:
    """
    Save the user's query and the two agent answers
    into answers.txt.
    """

    file_path = Path("answers.txt")

    content = f"""
============================================================
CUSTOMER SUPPORT BUILDATHON
============================================================

USER QUERY
------------------------------------------------------------
{query}

ASSISTANT ANSWER
------------------------------------------------------------
{answer_1}

WEB SEARCH ASSISTANT ANSWER
------------------------------------------------------------
{answer_2}

============================================================
"""

    file_path.write_text(content, encoding="utf-8")

    return f"Answers successfully saved to {file_path}"


# ============================================================
# 4. WEB SEARCH TOOL
# ============================================================

search_tool = SerperDevTool()


# ============================================================
# 5. CREATE AGENT 1 - ASSISTANT
# ============================================================

assistant_agent = Agent(
    role="Assistant",
    goal="Answer the user's query directly using your own knowledge.",
    backstory=(
        "You are the first customer support assistant. "
        "You answer the user's question directly and clearly "
        "using your own knowledge. Do not perform web searches."
    ),
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 6. CREATE AGENT 2 - WEB SEARCH ASSISTANT
# ============================================================

web_search_agent = Agent(
    role="Web Search Assistant",
    goal=(
        "Search the web for the user's query and provide "
        "a useful answer based on the search results."
    ),
    backstory=(
        "You are the second customer support assistant. "
        "You specialize in finding current information on the web. "
        "Use the web search tool to research the user's query "
        "and then provide a clear answer."
    ),
    tools=[search_tool],
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 7. CREATE AGENT 3 - ENTRY AGENT
# ============================================================

entry_agent = Agent(
    role="Entry Agent",
    goal=(
        "Save the user's query and the answers produced by the "
        "first two agents into a text file."
    ),
    backstory=(
        "You are the final agent in the customer support workflow. "
        "You receive the original query and the results from the "
        "previous two agents. You must save all three pieces of "
        "information into answers.txt."
    ),
    tools=[save_answers_to_file],
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Customer Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Customer Support")
st.write(
    "CrewAI + Streamlit Buildathon | "
    "Three sequential AI agents"
)

st.divider()

query = st.text_area(
    "Enter your query or task",
    placeholder="Example: How do I reset my password?",
    height=120
)


# ============================================================
# 9. RUN THE CREW
# ============================================================

if st.button("🚀 Submit Query", type="primary"):

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Running the three-agent workflow..."):

        # ----------------------------------------------------
        # Task 1 - Assistant
        # ----------------------------------------------------

        task_1 = Task(
            description=f"""
            Answer the following user query directly from your
            own knowledge.

            User Query:
            {query}

            Do not use web search.
            Provide a clear and useful answer.
            """,
            expected_output=(
                "A clear and direct answer to the user's query."
            ),
            agent=assistant_agent
        )

        # ----------------------------------------------------
        # Task 2 - Web Search Assistant
        # ----------------------------------------------------

        task_2 = Task(
            description=f"""
            Search the web for the following user query and
            provide an answer based on the search results.

            User Query:
            {query}

            Use the available web search tool.
            Provide a clear and useful answer.
            """,
            expected_output=(
                "A web-researched answer to the user's query."
            ),
            agent=web_search_agent,
            context=[task_1]
        )

        # ----------------------------------------------------
        # Task 3 - Entry Agent
        # ----------------------------------------------------

        task_3 = Task(
            description=f"""
            You are the final Entry Agent.

            Save the following information into answers.txt:

            USER QUERY:
            {query}

            The first agent's answer is available from the
            previous task.

            The second agent's web-search answer is available
            from the previous task.

            Use the save_answers_to_file tool to save:

            1. The original user query
            2. Answer from the Assistant
            3. Answer from the Web Search Assistant

            After saving the file, confirm that the information
            was successfully saved.
            """,
            expected_output=(
                "Confirmation that the query and both answers "
                "were saved to answers.txt."
            ),
            agent=entry_agent,
            context=[task_1, task_2]
        )

        # ----------------------------------------------------
        # Sequential Crew
        # ----------------------------------------------------

        crew = Crew(
            agents=[
                assistant_agent,
                web_search_agent,
                entry_agent
            ],
            tasks=[
                task_1,
                task_2,
                task_3
            ],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()


    # ========================================================
    # 10. DISPLAY RESULTS
    # ========================================================

    st.success("Three-agent workflow completed successfully!")

    st.divider()

    st.subheader("📌 User Query")

    st.write(query)

    st.divider()

    st.subheader("🤖 Assistant Answer")

    st.write(task_1.output.raw)

    st.divider()

    st.subheader("🌐 Web Search Assistant Answer")

    st.write(task_2.output.raw)

    st.divider()

    st.subheader("💾 Entry Agent")

    st.write(
        "The query and both answers have been saved to "
        "`answers.txt`."
    )