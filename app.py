
import os
import streamlit as st

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from pydantic import Field


# ----------------------------------------
# 1. Streamlit page configuration
# ----------------------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research Agent")
st.write(
    "Enter a research topic. Your AI agent will search the web, "
    "analyze information, and generate a research report."
)


# ----------------------------------------
# 2. Custom DuckDuckGo search tool
# ----------------------------------------
class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the public web using DuckDuckGo. "
        "Input a focused search query. Returns titles, URLs, "
        "and snippets from relevant search results."
    )

    max_results: int = Field(default=5)

    def _run(self, query: str) -> str:
        """Search the web and return source information."""

        try:
            with DDGS() as search:
                results = list(
                    search.text(
                        query,
                        max_results=self.max_results,
                    )
                )

            if not results:
                return "No search results found. Try a different query."

            formatted_results = []

            for index, result in enumerate(results, start=1):
                title = result.get("title", "Untitled")
                url = result.get("href", "")
                snippet = result.get("body", "")

                formatted_results.append(
                    f"Source {index}: {title}\n"
                    f"URL: {url}\n"
                    f"Snippet: {snippet}"
                )

            return "\n\n".join(formatted_results)

        except Exception as exc:
            return f"Search failed: {type(exc).__name__}: {exc}"


# ----------------------------------------
# 3. Read the Groq API key
# ----------------------------------------
def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("GROQ_API_KEY", "")


# ----------------------------------------
# 4. Create the research agent
# ----------------------------------------

def create_research_agent(api_key: str):
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
    )

    search_tool = DuckDuckGoSearchTool(max_results=5)

    researcher = Agent(
        role="Professional Research Analyst",
        goal=(
            "Research the user's topic using reliable web sources, "
            "compare findings, identify limitations, and write "
            "an accurate, well-structured research report."
        ),
        backstory=(
            "You are a careful research analyst. You distinguish "
            "facts from opinions, avoid unsupported claims, "
            "and never invent sources, quotations, statistics, "
            "or publication dates."
        ),
        llm=llm,
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
        max_iter=8,
    )

    return researcher


# ----------------------------------------
# 5. Define the research task
# ----------------------------------------
def create_research_task(agent, topic: str):
    return Task(
        description="""
        Research the following topic:

        {topic}

        Instructions:

        1. Break the topic into useful research questions.
        2. Use DuckDuckGo to search for relevant web sources.
        3. Search more than once using different focused queries
           when additional information is needed.
        4. Prefer primary sources, official reports, academic
           publications, and reputable organizations.
        5. Compare the information found across sources.
        6. Separate established facts from opinions and estimates.
        7. Do not invent facts, citations, statistics, or URLs.
        8. If evidence is insufficient or conflicting, explain this.
        9. Use source links from actual search results.
        10. Write a useful report based on the evidence collected.

        Use this report structure:

        # Research Report: [Topic]

        ## 1. Executive Summary
        ## 2. Introduction and Background
        ## 3. Research Questions and Approach
        ## 4. Key Findings
        ## 5. Detailed Analysis
        ## 6. Challenges, Risks, or Limitations
        ## 7. Conclusion and Recommendations
        ## 8. References

        Cite sources in the report using numbered references such
        as [1] and [2]. Include the corresponding source title
        and actual URL in the References section.

        Only cite sources that you actually received from the
        search tool. Never claim that a source supports a finding
        unless the available source information supports it.

        Be clear, professional, and informative. Avoid unnecessary
        repetition. Do not present this report as peer-reviewed
        research unless it has actually been peer-reviewed.
        """,

        expected_output=(
            "A complete Markdown research report with an executive "
            "summary, evidence-based findings, analysis, conclusion, "
            "recommendations, and numbered references with source URLs."
        ),

        agent=agent,
    )


# ----------------------------------------
# 6. Research topic input
# ----------------------------------------
topic = st.text_area(
    "Enter your research topic",
    placeholder=(
        "Example: The impact of artificial intelligence "
        "on accounting jobs in Pakistan"
    ),
    height=120,
)

generate_button = st.button(
    "Generate Research Report",
    type="primary",
    use_container_width=True,
)


# ----------------------------------------
# 7. Run the AI research agent
# ----------------------------------------
if generate_button:
    if not topic.strip():
        st.warning("Please enter a research topic.")

    elif len(topic.strip()) < 5:
        st.warning("Please enter a more descriptive research topic.")

    else:
        api_key = get_groq_api_key()

        if not api_key:
            st.error(
                "Groq API key not found. Add GROQ_API_KEY "
                "to Streamlit Secrets."
            )
            st.stop()

        try:
            with st.spinner(
                "The AI agent is searching, analyzing, and writing..."
            ):
                agent = create_research_agent(api_key)

                task = create_research_task(
                    agent,
                    topic.strip(),
                )

                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )

                result = crew.kickoff(
                    inputs={"topic": topic.strip()}
                )

            report = result.raw

            st.success("Research report generated successfully!")

            st.markdown(report)

            st.download_button(
                label="Download Report as Markdown",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(
                "The research process failed. Check your API key, "
                "dependency versions, search connectivity, and "
                "deployment logs."
            )
            st.caption(f"Error type: {type(exc).__name__}")
            st.exception(exc)
