"""arxiv-reader: A LangGraph application that reads a digest of arXiv papers and returns 
most relevant papers to a set of topics."""

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")


class Paper(BaseModel):
    title:str
    authors:list[str]
    link:str
    abstract:str
    relevance_score:float
    relevance_explanation:str

class GraphState(TypedDict):
    topic: str
    digest: str
    report: str
    papers: list[Paper]

class RetrieverOutput(BaseModel):
    papers: list[Paper] = Field(description="List of relevant papers for the given topic.")

model_retriever = model.with_structured_output(RetrieverOutput)

def start_state(state: MessagesState) -> GraphState:
    """Initialize the retriever state with topic and digest."""
    return {"topic": "Task-oriented conversational agents", "digest": state["messages"][-1].content}

def retrieve_per_topic(state: GraphState) -> GraphState:
    """Retrieve relevant arXiv papers for a given topic from a digest."""
    topic = state["topic"]
    prompt = f"""
    You are an expert research assistant. Given the following digest of arXiv papers, 
    identify and return the most relevant papers for the topic: "{topic}".

    Digest:
    {state['digest']}

    For each relevant paper, provide the following details:
    - Title
    - Authors (as a list)
    - Link
    - Abstract
    - Publication - publication venue if available
    - Relevance Score (0-100)
    - Explanation of Relevance

    Return the results in JSON format as a list of papers. Return only the papers that are listed in the digest.
    """
    return {"papers": model_retriever.invoke([HumanMessage(prompt)]).papers}

def filter_relevant_papers(state: GraphState) -> GraphState:
    """Filter papers based on a relevance score threshold."""
    # Keep papers whose link is found in the digest
    filtered_papers = [paper for paper in state["papers"] if paper.link in state["digest"]]

    # Filter papers with relevance score above a threshold (e.g., 50)
    filtered_papers = [paper for paper in filtered_papers if paper.relevance_score >= 50]

    # Sort papers by relevance score in descending order
    sorted_papers = sorted(filtered_papers, key=lambda x: x.relevance_score, reverse=True)
    return {"papers": sorted_papers}

def finalize_report(state: GraphState) -> GraphState:
    """Finalize the report by summarizing the retrieved papers."""
    prompt = f"""
    Given this topic: {state["topic"]},

    You have retrieved the following relevant papers, sorted by relevance:

    {state["papers"]}

    Summarize the key findings and insights from these papers in relation to the topic. Include references to the relevant papers by inserting footnotes as indicated below:

    text text text [1], text text text [2], etc.

    1. Author et al., "Paper Title", publication Venue, Year. Link: URL
    2. Author et al., "Paper Title", publication Venue, Year. Link: URL
    3. ...
    """
    return {"report": model.invoke([HumanMessage(prompt)]).content}


workflow = StateGraph(MessagesState)
workflow.add_node("start", start_state)
workflow.add_node("retriever", retrieve_per_topic)
workflow.add_node("filter", filter_relevant_papers)
workflow.add_node("report", finalize_report)
workflow.add_edge(START, "start")
workflow.add_edge("start", "retriever")
workflow.add_edge("retriever", "filter")
workflow.add_edge("filter", "report")
workflow.add_edge("report", END)

graph = workflow.compile()

