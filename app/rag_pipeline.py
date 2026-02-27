from app.vector_store import search
from app.llm import generate_answer, rewrite_question
from app.memory import add_to_memory, format_chat_history


def rag_pipeline(question: str, session_id: str = None):

    print("---- RAG PIPELINE START ----")
    print("Incoming Question:", question)
    print("Session ID:", session_id)

    if not session_id:
        session_id = "temp"

    chat_history_text = format_chat_history(session_id)

    print("Chat History:", chat_history_text)

    if chat_history_text:
        standalone_question = rewrite_question(question, chat_history_text)
    else:
        standalone_question = question

    print("Standalone Question:", standalone_question)

    docs, confidence = search(standalone_question, k=3)
    print("Confidence:", confidence)
    print("Docs Retrieved:", len(docs))

    # HARD FILTER
    if confidence < 0.40:
        return {
            "answer": "Information not available in knowledge base.",
            "confidence": round(confidence, 3)
        }

    context = "\n".join(docs)

    answer = generate_answer(standalone_question, context)

    add_to_memory(session_id, question, answer)
    
    return {
        "answer": answer,
        "confidence": round(confidence, 3)
    }