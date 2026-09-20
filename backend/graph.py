from langgraph.graph import StateGraph, END
from embedder import embed_text
from openrouter_client import generate_answer
from db import SessionLocal
from vector_store import similarity_search


class GraphState(dict):
    user_id: str
    question: str
    context: str
    answer: str


def retrieve(state: GraphState):
    db = SessionLocal()
    try:
        embedding = embed_text(state["question"])
        docs = similarity_search(db, state["user_id"], embedding, k=5)
        state["context"] = docs
        return state
    finally:
        db.close()


def generate(state: GraphState):
    context_text = "\n\n".join(state.get("context") or [])
    state["answer"] = generate_answer(state["question"], context_text)
    return state


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()
