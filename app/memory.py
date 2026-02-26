# Store memory per session
session_memory = {}

def get_history(session_id: str):
    return session_memory.get(session_id, [])

def add_to_memory(session_id: str, question: str, answer: str):
    if session_id not in session_memory:
        session_memory[session_id] = []

    session_memory[session_id].append({
        "question": question,
        "answer": answer
    })

def format_chat_history(session_id: str):
    history = get_history(session_id)
    messages = []

    for item in history[-3:]:  # last 3 turns only
        messages.append(f"User: {item['question']}")
        messages.append(f"Assistant: {item['answer']}")

    return "\n".join(messages)

def clear_session(session_id: str):
    session_memory.pop(session_id, None)