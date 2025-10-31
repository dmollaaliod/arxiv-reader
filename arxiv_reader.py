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
    return {"topic": "Task-oriented conversational agents", "digest": state["messages"][0].content}

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
    - Relevance Score (0-100)
    - Explanation of Relevance

    Return the results in JSON format as a list of papers, sorted by relevance score in descending order.
    """
    return {"papers": model_retriever.invoke([HumanMessage(prompt)])}

def finalize_report(state: GraphState) -> GraphState:
    """Finalize the report by summarizing the retrieved papers."""
    prompt = f"""
    Given this topic: {state["topic"]},

    You have retrieved the following relevant papers, sorted by relevance:

    {state["papers"]}

    Summarize the key findings and insights from these papers in relation to the topic.
    """
    return {"report": model.invoke([HumanMessage(prompt)]).content}


workflow = StateGraph(MessagesState)
workflow.add_node("start", start_state)
workflow.add_node("retriever", retrieve_per_topic)
workflow.add_node("report", finalize_report)
workflow.add_edge(START, "start")
workflow.add_edge("start", "retriever")
workflow.add_edge("retriever", "report")
workflow.add_edge("report", END)

graph = workflow.compile()

