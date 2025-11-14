from crewai import Agent
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun


class DuckSearchTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Busca información en la web usando DuckDuckGo."

    def _run(self, query: str):
        duck = DuckDuckGoSearchRun()
        return duck.run(query)

# ============================================================
#   AGENTES
# ============================================================

def get_agents():
    
    search_tool = DuckSearchTool()

    researcher = Agent(
        name="Researcher",
        role="Web Research Specialist",
        goal="Buscar información confiable en la web.",
        backstory="Experto en búsqueda avanzada de información pública.",
        tools=[search_tool],
        llm="HuggingFaceH4/zephyr-7b-beta"
    )

    writer = Agent(
        name="Writer",
        role="Research Writer",
        goal="Escribir un resumen coherente y claro.",
        backstory="Especialista en síntesis y redacción.",
        llm="HuggingFaceH4/zephyr-7b-beta"
    )

    reviewer = Agent(
        name="Reviewer",
        role="Quality Reviewer",
        goal="Verificar precisión y consistencia.",
        backstory="Editor experto en fact-checking.",
        llm="microsoft/deberta-v3-small"
    )

    return researcher, writer, reviewer